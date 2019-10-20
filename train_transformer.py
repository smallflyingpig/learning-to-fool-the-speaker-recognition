import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np 
import logging
import tqdm
import random
from functools import partial
from tensorboardX import SummaryWriter
from optparse import OptionParser
from dataset import TIMIT_speaker
from model import SincClassifier
from transformer import SpeechTransformer
from utils import read_conf, get_dict_from_args
from trainer import ClassifierTrainer, save_checkpoint, load_checkpoint
from copy import deepcopy
import multiprocessing
import pandas as pd 
multiprocessing.set_start_method('spawn', True)


def get_optimizer(opt_type, params, lr, **kwargs):
    opt_dict = {'sgd':optim.SGD, 'rmsprop':optim.RMSprop}
    return opt_dict[opt_type](params, lr=lr, **kwargs)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

@torch.no_grad()
def evaluate(model, test_dataset, cost):
    test_dataloader = DataLoader(test_dataset, 128, shuffle=False, num_workers=8, pin_memory=True)
    model.eval()
    bar = tqdm.tqdm(test_dataloader)
    loss_all = {}
    err_fake_all = {}
    noise_all = {}
    for idx, data in enumerate(bar):
        wav_data, speaker_id, phoneme = data
        wav_data, speaker_id, phoneme = wav_data.float().cuda(), speaker_id.long().cuda(), phoneme.long().cuda()
        pout = model.forward(wav_data)
        labels = {"speaker":speaker_id, "speech":wav_data, "norm":wav_data}
        loss_func = cost
        loss_total, loss_dict, loss_dict_grad, pred_dict = loss_func(pout, labels)
        pred = torch.max(pred_dict['speaker'], dim=1)[1]
        err_speaker = torch.mean((pred != speaker_id).float()).detach().cpu().item()
        
        pred = torch.max(pred_dict['speech'], dim=1)[1]
        err_speech = torch.mean((pred != phoneme).float()).detach().cpu().item()
        err_dict = {"err_spk":err_speaker, "err_sph":err_speech}
        err_str = get_dict_str(err_dict)

        loss_total = loss_total.detach().cpu().item()

        noise = (pout.detach()-wav_data.detach())
        noise_mean, noise_std, noise_abs = torch.mean(noise).item(), torch.std(noise).item(), torch.mean(torch.abs(noise)).item()
        noise_dict = {"mean":noise_mean*1e3, "std":noise_std*1e3, "m_abs":noise_abs*1e3}
        noise_str = get_dict_str(noise_dict)
        def accumulate_dict(total_dict, item_dict, factor):
            for k,v in item_dict.items():
                total_dict[k] = total_dict.get(k,0)+v*factor
            return total_dict
        loss_all = accumulate_dict(loss_all, loss_dict, len(phoneme))
        err_fake_all = accumulate_dict(err_fake_all, err_dict, len(phoneme))
        noise_all = accumulate_dict(noise_all, noise_dict, len(phoneme))
        
        bar.set_description("err:({}), noise(e-3):({}), batch size:{}".format(err_str, noise_str, len(phoneme)))
    bar.close()
    def multiply_dict(data_dict, factor):
        for k,v in data_dict.items():
            data_dict[k] = v*factor
        return data_dict
    loss_all = multiply_dict(loss_all, 1.0/len(test_dataset))
    err_fake_all = multiply_dict(err_fake_all, 1.0/len(test_dataset))
    noise_all = multiply_dict(noise_all, 1.0/len(test_dataset))
    print(get_dict_str(loss_all), get_dict_str(err_fake_all), get_dict_str(noise_all))

@torch.no_grad()
def sentence_test(speaker_model, wav_data, wlen=3200, wshift=10, batch_size=128):
    """
    wav_data: B, L
    """
    wav_data = wav_data.squeeze()
    L = wav_data.shape[0]
    pred_all = []
    begin_idx = 0
    batch_data = []
    while begin_idx<L-wlen:
        batch_data.append(wav_data[begin_idx:begin_idx+wlen])
        if len(batch_data)>=batch_size:
            pred_batch = speaker_model(torch.stack(batch_data))
            pred_all.append(pred_batch)
            batch_data = []
        begin_idx += wshift
    if len(batch_data)>0:
        pred_batch = speaker_model(torch.stack(batch_data))
        pred_all.append(pred_batch)
    [val,best_class]=torch.max(torch.sum(torch.cat(pred_all, dim=0),dim=0),0)
    return best_class.detach().cpu().item()
    

import soundfile as sf
from utils import SNR, PESQ
from trainer import RunningAverage
@torch.no_grad()
def test_wav(model, filename_list, data_folder, out_folder, speaker_model=None, label_dict=None, target=-1):
    model.eval()
    if speaker_model: speaker_model.eval()
    bar = tqdm.tqdm(filename_list)
    averager = RunningAverage()
    pertutations = []
    pred_results = []
    for idx, filename in enumerate(bar):
        real_data, fs = sf.read(os.path.join(data_folder, filename))
        real_data_norm, real_norm_factor = TIMIT_speaker.preprocess(real_data)
        fake_data_norm = model(torch.from_numpy(real_data_norm).float().cuda().unsqueeze(0)).squeeze().detach().cpu().numpy()

        fake_data = fake_data_norm*real_norm_factor
        # save data
        output_filename = os.path.join(out_folder, filename)
        if not os.path.exists(os.path.dirname(output_filename)): 
            os.makedirs(os.path.dirname(output_filename))
        # print(fake_data.shape)
        sf.write(output_filename, fake_data, fs)
        snr = SNR(fake_data, real_data)
        pesq = PESQ(fake_data, real_data, fs)
        averager.update({"SNR":snr, "PESQ":pesq}, {"SNR":snr, "PESQ":pesq})
        output_str = "SNR:{:5.2f}, PESQ:{:5.2f}".format(snr, pesq)
        pertutations.append((real_data-fake_data).astype(np.float16))
        if speaker_model:
            label = label_dict[filename]
            pred_fake = sentence_test(speaker_model, torch.from_numpy(fake_data_norm).float().cuda().unsqueeze(0))
            if target != -1:
                err_rate = (pred_fake == target)
                averager.update({"err_rate":err_rate}, {"err_rate":1})
                if label == target:
                    pred_real = sentence_test(speaker_model, torch.from_numpy(real_data_norm).float().cuda().unsqueeze(0))
                    averager.update({"err_rate_raw":pred_real!=label, "target_rate_raw":pred_real==target}, {"err_rate_raw":1, "target_rate_raw":1})

                pred_results.append({'file':filename, 'pred_real':pred_real, 'pred_fake':pred_fake, 'label':label})
            else:
                err_rate = (pred_fake != label)
                averager.update({"err_rate":err_rate}, {"err_rate":1})
                pred_results.append({'file':filename, 'pred_fake':pred_fake, 'label':label})
            output_str += ", real/fake:{}/{}, data len:{}".format(label, pred_fake, fake_data.shape)
        bar.set_description(output_str)
    np.save(os.path.join(out_folder, "pertutation.npy"), (pertutations))
    if len(pred_results)>0:
        pd.DataFrame(pred_results).to_csv(os.path.join(out_folder, "pred_results.csv"))
    bar.close()
    avg = averager.average()
    print(get_dict_str(avg))


import time
@torch.no_grad()
def test_wav_cpu(model, filename_list, data_folder, out_folder, speaker_model=None, label_dict=None, target=-1):
    model = model.cpu().eval()
    if speaker_model: speaker_model.eval()
    bar = tqdm.tqdm(filename_list)
    averager = RunningAverage()
    for idx, filename in enumerate(bar):
        real_data, fs = sf.read(os.path.join(data_folder, filename))
        time_data = len(real_data)/float(fs)
        time_start = time.time()
        real_data_norm, real_norm_factor = TIMIT_speaker.preprocess(real_data)
        fake_data_norm = model(torch.from_numpy(real_data_norm).float().unsqueeze(0)).squeeze().detach().cpu().numpy()
        time_end = time.time()
        pro_time = time_end-time_start
        ratio = pro_time/time_data
        averager.update({"ratio":ratio}, {"ratio":1})
        bar.set_description("ratio:{:5.3f}[{:5.3f}/{:5.3f}]".format(ratio, pro_time, time_data))
    bar.close()
    avg = averager.average()
    print(get_dict_str(avg))

grads = {}
def save_grad(v):
    def hook(grad):
        grads[v] = grad
    return hook

def get_dict_str(d):
    s = ','.join(["{}:{:5.3f}".format(k,v) for k,v in d.items()])
    return s


def batch_process_speaker(model:SincClassifier, data, train_mode=True, **kwargs)->dict:
    wav_data, speaker_id, phoneme, norm_factor = data
    wav_data, speaker_id, phoneme, norm_factor = wav_data.float().cuda().requires_grad_(), speaker_id.long().cuda(), phoneme.long().cuda(), norm_factor.float().cuda()
    if train_mode:
        model.train()
        optimizer, loss_func = kwargs.get('optimizer'), kwargs.get('loss_func')
        pout = model.forward(wav_data)
        # pout = pout_raw/torch.max(torch.abs(pout_raw), dim=1)[0].unsqueeze(1) # normlize
        # pout = pout.detach().requires_grad_()
        pout.register_hook(save_grad("wav_data"))
        labels = {"speaker":speaker_id, "speech":wav_data, "norm":wav_data}
        loss_total, loss_dict, loss_dict_grad, pred_dict = loss_func(pout, labels)
        grad_dict = {}
        for k, l in loss_dict_grad.items():
            model.zero_grad()
            l.backward(retain_graph=True)
            grad_dict[k] = grads['wav_data'].abs().mean()
        model.zero_grad()
        loss_total.backward()
        grad_dict['total'] = grads['wav_data'].abs().mean()
        optimizer.step()
    

        pred = torch.max(pred_dict['speaker'], dim=1)[1]
        err_speaker = torch.mean((pred != speaker_id).float()).detach().cpu().item()
        
        pred = torch.max(pred_dict['speech'], dim=1)[1]
        err_speech = torch.mean((pred != phoneme).float()).detach().cpu().item()
        err_dict = {"err_spk":err_speaker, "err_sph":err_speech}
        err_str = get_dict_str(err_dict)

        loss_total = loss_total.detach().item()
        loss_str = get_dict_str(loss_dict)

        loss_dict.update(err_dict)

        noise = (pout.detach()-wav_data.detach())
        noise_mean, noise_std, noise_abs = torch.mean(noise).item(), torch.std(noise).item(), torch.mean(torch.abs(noise)).item()
        noise_dict = {"mean":noise_mean*1e3, "std":noise_std*1e3, "m_abs":noise_abs*1e3}
        noise_str = get_dict_str(noise_dict)
        loss_dict.update(noise_dict)
        grad_dict = {k:v*1e3 for k,v in grad_dict.items()}
        grad_str = get_dict_str(grad_dict)
        loss_dict.update(grad_dict)
        rtn = {
            "output":"loss_total:{:6.3f}({}), err:({}), lr(e-3):[{:6.3f}], grad(e-3):({}), noise(e-3):({})".format(
                 loss_total, loss_str, err_str, optimizer.param_groups[0]['lr']*1e3, grad_str, noise_str),
            "vars":loss_dict,
            "count":{k:len(speaker_id) for k in loss_dict}
        }
        
    else: #eval
        model.eval()
        optimizer, loss_func = kwargs.get('optimizer'), kwargs.get('loss_func')
        with torch.no_grad():
            pout = model.forward(wav_data)
            labels = {"speaker":speaker_id, "speech":wav_data, "norm":wav_data}
            loss_total, loss_dict, loss_dict_grad, pred_dict = loss_func(pout, labels)
        pred = torch.max(pred_dict['speaker'], dim=1)[1]
        err_speaker = torch.mean((pred != speaker_id).float()).detach().cpu().item()
        
        pred = torch.max(pred_dict['speech'], dim=1)[1]
        err_speech = torch.mean((pred != phoneme).float()).detach().cpu().item()
        err_dict = {"err_spk":err_speaker, "err_sph":err_speech}
        err_str = get_dict_str(err_dict)

        loss_total = loss_total.detach().cpu().item()
        loss_str = get_dict_str(loss_dict)

        loss_dict.update(err_dict)

        noise = (pout.detach()-wav_data.detach())
        noise_mean, noise_std, noise_abs = torch.mean(noise).item(), torch.std(noise).item(), torch.mean(torch.abs(noise)).item()
        noise_dict = {"mean":noise_mean*1e3, "std":noise_std*1e3, "m_abs":noise_abs*1e3}
        noise_str = get_dict_str(noise_dict)
        loss_dict.update(noise_dict)
        rtn = {
            "output":"loss:{:6.3f}({}), err:({}), noise(e-3):({})".format(
                 loss_total, loss_str, err_str, noise_str),
            "vars":loss_dict,
            "count":{k:len(speaker_id) for k in loss_dict}
        }
    return rtn

class EvalHook(object):
    def __init__(self):
        self.best_accu = 0
    
    def __call__(self, model:nn.Module, epoch_idx, output_dir, 
        eval_rtn:dict, test_rtn:dict, logger:logging.Logger, writer:SummaryWriter):
        # save model
        acc = eval_rtn.get('err_spk', 0)-eval_rtn.get('err_sph', 1)
        is_best = acc > self.best_accu
        self.best_accu = acc if is_best else self.best_accu
        model_filename = "epoch_{}.pth".format(epoch_idx)
        save_checkpoint(model, os.path.join(output_dir, model_filename), 
            meta={'epoch':epoch_idx})
        os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "latest.pth"))
            )
        if is_best:
            os.system(
            "ln -sf {} {}".format(os.path.abspath(os.path.join(output_dir, model_filename)), 
            os.path.join(output_dir, "best.pth"))
            )

        if logger is not None:
            logger.info("EvalHook: best accu: {:.3f}, is_best: {}".format(self.best_accu, is_best))


class AdversarialLoss(object):
    """
    loss_all: a dict include all loss
    loss: model, factor, loss_func
    """
    def __init__(self, loss_all):
        self.loss_all = loss_all

    def zero_grad_labels_model(self):
        for k,l in self.loss_all.items():
            model = l.get('model', None)
            if model is not None:
                model.zero_grad()
        return self.loss_all

    def __call__(self, pred, labels):
        loss_dict_grad = {}
        loss_dict = {}
        pred_dict = {}
        for key, loss in self.loss_all.items():
            model = loss.get('model', None)
            if model is not None:
                pred_this = model(pred)
            else:
                pred_this = pred
            loss_func, label_this = loss["loss_func"], labels[key]
            loss_this = loss_func(pred, label_this) * loss['factor']
            loss_dict[key] = loss_this.detach().cpu().item()
            loss_dict_grad[key] = loss_this
            pred_dict[key] = pred_this.detach()
        loss_list = [v for k,v in loss_dict_grad.items()]
        loss_total = sum(loss_list)
        # loss_total = loss_dict_grad['norm'] * self.loss_all['norm']['factor']
        loss_dict["loss_total"] = loss_total.detach().cpu().item()
        return loss_total, loss_dict, loss_dict_grad, pred_dict
            
class MaxRankLoss(object):
    def __init__(self, clip_abs=0.1):
        self.clip_abs = clip_abs

    def __call__(self, pred, label):
        """
        pred: (B,D)
        label: (B,)
        """
        B = len(label) 
        max_data, max_idx = pred.max(dim=1)
        select = (max_idx != label)

        loss = pred[torch.arange(B), label][select].sum()/select.sum()
        return loss


class SpeakerLoss(object):
    def __init__(self, model:SincClassifier, clamp=1, mul=0.01):
        self.model = model
        self.model.eval()
        self.clamp = clamp
        self.mul = mul

    def __call__(self, pred, label):
        B = len(label)
        # pred = F.softmax(self.model(pred), dim=1)
        pred = self.model(pred)

        max_data, max_idx = torch.topk(pred, k=2, dim=1)
        select = (max_idx[:,0].detach() == label)
        if select.sum() <1:
            loss = -pred[torch.arange(B), max_idx[:,1]][select].sum() # 0
        else:
            loss = -((pred[torch.arange(B), max_idx[:,1]][select]-pred[torch.arange(B), max_idx[:,0]][select])).mul_(self.mul).clamp_(-self.clamp, self.clamp).sum()/select.sum()
        return loss

class SpeakerLossTarget(object):
    def __init__(self, model:SincClassifier, target=0, clamp=1, mul=0.01):
        self.model = model
        self.target = int(target)
        self.model.eval()
        self.clamp = clamp
        self.mul = mul

    def __call__(self, pred, label):
        B = len(label)
        # pred = F.softmax(self.model(pred), dim=1)
        pred = self.model(pred)

        max_data, max_idx = torch.topk(pred, k=2, dim=1)
        select = (max_idx[:,0].detach() != self.target)
        if select.sum() <1:
            loss = -pred[torch.arange(B), max_idx[:,0]][select].sum() # 0
        else:
            loss = -((pred[:, self.target][select]-pred[torch.arange(B), max_idx[:,0]][select])).mul_(self.mul).clamp_(-self.clamp, self.clamp).sum()/select.sum()
        return loss


class SpeechLoss(object):
    def __init__(self, model:SincClassifier, factor_kld=10):
        self.model = model
        self.model.eval()
        self.kld = nn.KLDivLoss(reduction='mean')
        self.factor_kld = factor_kld

    def __call__(self, pred, label):
        B = label.shape[0]
        # dis_pred = F.softmax(self.model(pred), dim=1)
        # dis_label = F.softmax(self.model(label.requires_grad_(False)), dim=1)
        pred, pred_immediate = self.model(pred, immediate=True)
        label, label_immediate = self.model(label.requires_grad_(False), immediate=True)
        dis_pred = F.softmax(pred, dim=1)
        dis_label = F.softmax(label, dim=1)
        # _, max_pred_idx = dis_pred.max(dim=1)
        # _, max_label_idx = dis_label.max(dim=1)
        # select = (max_pred_idx.detach() != max_label_idx)
        # if select.sum()<1:
        #     loss_class = pred[torch.arange(B), max_pred_idx][select].sum() # 0
        # else:
        #     loss_class = -(pred[torch.arange(B), max_label_idx][select]-pred[torch.arange(B), max_pred_idx][select]).sum()/select.sum() # 0
        # # gt dis error
        # pred_err = label[torch.arange(B), max_pred_idx][select]-pred[torch.arange(B), max_label_idx][select]
        # select_tmp = pred_err>0
        # if select_tmp.sum()<1:
        #     loss_pred_err = pred_err[select_tmp].sum()
        # else:
        #     loss_pred_err = pred_err[select_tmp].mean()

        # loss_kld = torch.abs(pred_immediate-label_immediate).mean()
        loss_kld = self.kld(dis_pred.log(), dis_label.detach()) # (torch.abs(pred-label) * (dis_pred.detach()-dis_label.detach()).abs()).mean()
        # loss_kld = torch.pow((pred-label).mul_(0.1).clamp_(-1,1), 2).mean()*0.5
        # loss_kld = (pred-label).mean(dim=1)

        # loss_kld = compute_mmd(pred_immediate, label_immediate)
        # return loss_pred_err + loss_class + loss_kld*self.factor_kld
        return loss_kld*self.factor_kld

        # return loss_kld*self.factor_kld

class DeMaxRankLoss(object):
    def __init__(self, clip_abs=0.1):
        self.clip_abs = clip_abs

    def __call__(self, pred, label):
        """
        pred: (B,D)
        label: (B,)
        """
        B = len(label)
        max_data, max_idx = pred.max(dim=1)
        select = (max_idx == label)

        loss = pred[torch.arange(B), label][select].sum()/select.sum()
        return loss

class MSEWithThreshold(object):
    
    def __init__(self, threshold=0.05, order=2):
        norm = {1:self.l1_with_threshold, 2:self.l2_with_threshold}
        self.threshold = threshold
        self.norm = norm[order]

    def l1_with_threshold(self, err):
        err = torch.abs(err)
        err = err-self.threshold
        err[err<0] = 0
        select_count = (err>0).sum()
        if select_count == 0:
            loss = err.sum()
        else:
            loss = err.sum()/select_count
        return loss
    
    def l2_with_threshold(self, err):
        err = err*err
        err = err-self.threshold*self.threshold
        err[err<0] = 0
        select_count = (err>0).sum()
        if select_count == 0:
            loss = err.sum()
        else:
            loss = err.sum()/select_count
        return loss

    def __call__(self, pred, label):
        err = pred-label
        return self.norm(err)
        
def get_option():
    parser=OptionParser()
    parser.add_option("--local_rank", type=int, default=0)
    parser.add_option("--no_dist", action='store_true', default=False)
    parser.add_option("--eval", action='store_true', default=False, help="eval the model")
    parser.add_option("--test", action='store_true', default=False, help="test the model")
    parser.add_option("--pt_file", type=str, default='none', help="path for pretrained file")
    parser.add_option("--data_root", type=str, default='./data/TIMIT/TIMIT_lower', help="path for data")
    parser.add_option("--output_dir", type=str, default="./output/timit_transformer")
    parser.add_option("--dataset", choices=['timit', 'libri'], default='timit', help="the dataset name")
    parser.add_option("--num_workers", type=int, default=8, help="num workers for dataloader")
    parser.add_option("--speaker_model", type=str, default="./output/timit_speaker/epoch_23.pth", help="path for pretrained speaker model")
    parser.add_option("--speech_model", type=str, default="./output/timit_speech/epoch_23.pth", help="path for pretrained speech model")
    parser.add_option("--channel", type=int, nargs='+', default=[32,32,32,32,32], help="channel for tranformer model")
    parser.add_option("--kernel_size", type=int ,nargs='+', default=[3,3,3,3,3], help="kernel size for transformer model")
    parser.add_option("--dilation", type=int, nargs='+', default=[1,2,5,2,1], help="dilation for transformer model")
    parser.add_option("--sample", type=int, nargs='+', default=[1,1,1,1,1], help="sample for transformer model")
    parser.add_option("--noise_scale", type=float, default=1.0, help="scele for the noise")
    parser.add_option("--speaker_factor", type=float, default=10.0, help="factor for speaker loss")
    parser.add_option("--speech_factor", type=float, default=10.0, help="factor for speech loss")
    parser.add_option("--norm_factor", type=float, default=1000.0, help="factor for l1 norm")
    parser.add_option("--speech_kld_factor", type=float, default=1.0, help="factor for speech kld")
    parser.add_option("--norm_clip", type=float, default=0.05, help="clip for the norm loss")
    parser.add_option("--target", type=int, default=-1, help="target attack label")
    parser.add_option("--speaker_cfg", type=str, default="./config/timit_speaker_transformer.cfg", help="")
    parser.add_option("--speech_cfg", type=str, default="./config/timit_speech.cfg")
    parser.add_option("--cpu_test", action='store_true', default=False, help="test the cpu time")
    (args,_)=parser.parse_args()
    return args


def get_pretrained_models(args_speaker, args_speech):
    args_all = {"speaker":args_speaker, "speech":args_speech}
    models = {}
    for key, args in args_all.items():
        CNN_arch = get_dict_from_args(['cnn_input_dim', 'cnn_N_filt', 'cnn_len_filt','cnn_max_pool_len',
                  'cnn_use_laynorm_inp','cnn_use_batchnorm_inp','cnn_use_laynorm','cnn_use_batchnorm',
                  'cnn_act','cnn_drop'], args.cnn)
    
        DNN_arch = get_dict_from_args(['fc_input_dim','fc_lay','fc_drop',
                'fc_use_batchnorm','fc_use_laynorm','fc_use_laynorm_inp','fc_use_batchnorm_inp',
                'fc_act'], args.dnn)
    
        Classifier = get_dict_from_args(['fc_input_dim','fc_lay','fc_drop', 
                  'fc_use_batchnorm','fc_use_laynorm','fc_use_laynorm_inp','fc_use_batchnorm_inp',
                  'fc_act'], args.classifier)
    
        CNN_arch['fs'] = args.windowing.fs
        model = SincClassifier(CNN_arch, DNN_arch, Classifier)
        if args.model_path!='none':
            print("load model from:", args.model_path)
            if os.path.splitext(args.model_path)[1] == '.pkl':
                checkpoint_load = torch.load(args.model_path)
                model.load_raw_state_dict(checkpoint_load)
            else:
                load_checkpoint(model, args.model_path, strict=True)

        model = model.cuda().eval()
        # freeze the model
        for p in model.parameters():
            p.requires_grad = False
        models[key] = model
        
    return models

def _init_fn(work_id, seed):
    np.random.seed(seed+work_id)

def main(args):
    speaker_cfg = args.speaker_cfg
    speech_cfg = args.speech_cfg
    args_speaker = read_conf(speaker_cfg, deepcopy(args))
    args_speaker.model_path = args.speaker_model
    args_speech = read_conf(speech_cfg, deepcopy(args))
    args_speech.model_path = args.speech_model
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("set seed: ", args_speaker.optimization.seed)
    torch.manual_seed(args_speaker.optimization.seed)
    np.random.seed(args_speaker.optimization.seed)
    random.seed(args_speaker.optimization.seed)

    torch.cuda.set_device(args.local_rank)
    if not args.no_dist:
        torch.distributed.init_process_group(backend="nccl")

    train_dataset = TIMIT_speaker(args.data_root, train=True, phoneme=True, norm_factor=True)
    test_dataset = TIMIT_speaker(args.data_root, train=False, phoneme=True, norm_factor=True)
    
    pretrained_models = get_pretrained_models(args_speaker, args_speech)

    loss_factors = {"speaker":args.speaker_factor, "speech":args.speech_factor, "norm":args.norm_factor}
    if args.target < 0: # non-targeted
        speaker_loss = SpeakerLoss(pretrained_models['speaker'])
    else: # targeted attack
        speaker_loss = SpeakerLossTarget(pretrained_models['speaker'], args.target)
    loss_all = {}
    loss_all['speech'] = {'model':pretrained_models['speech'], 'factor':loss_factors['speech'], 'loss_func':SpeechLoss(pretrained_models['speech'], factor_kld=args.speech_kld_factor)}
    loss_all['speaker'] = {'model':pretrained_models['speaker'], 'factor':loss_factors['speaker'], 'loss_func':speaker_loss}
    loss_all['norm'] = {'loss_func':MSEWithThreshold(args.norm_clip), 'factor':loss_factors['norm']}
    
    cost = AdversarialLoss(loss_all)

    model = SpeechTransformer(args.channel, args.kernel_size, args.dilation, args.sample, args.noise_scale)

    if args.pt_file!='none':
        print("load model from:", args.pt_file)
        if os.path.splitext(args.pt_file)[1] == '.pkl':
            checkpoint_load = torch.load(args.pt_file)
            model.load_raw_state_dict(checkpoint_load)
        else:
            load_checkpoint(model, args.pt_file)
        
    model = model.cuda()
    if args.eval:
        assert args.pt_file != 'none', "no pretrained model is provided!"
        print('only eval the model')
        evaluate(model, test_dataset, cost)
        return
    if args.test:
        assert args.pt_file != 'none', "no pretrained model is provided!"
        print("only test the model")
        filename_list = open("./data/TIMIT/speaker/test.scp", 'r').readlines()
        filename_list = [_f.strip() for _f in filename_list]
        label_dict = np.load(os.path.join(args.data_root, "processed", "TIMIT_labels.npy")).item()
        test_wav(model, filename_list, args.data_root, os.path.join(args.data_root, "output"), pretrained_models['speaker'], label_dict, args.target)
        return
    if args.cpu_test:
        assert args.pt_file != 'none', "no pretrained model is provided!"
        print("only cpu test the model")
        filename_list = open("./data/TIMIT/speaker/test.scp", 'r').readlines()
        filename_list = [_f.strip() for _f in filename_list]
        label_dict = np.load(os.path.join(args.data_root, "processed", "TIMIT_labels.npy")).item()
        test_wav_cpu(model, filename_list, args.data_root, os.path.join(args.data_root, "output"), pretrained_models['speaker'], label_dict, args.target)
        return

    print("train the model")
    batch_process = batch_process_speaker
    eval_hook = EvalHook()
    optimizer = optim.Adam(model.parameters(), lr=args_speaker.optimization.lr, betas=(0.95, 0.999))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
    if args.no_dist:
        kwarg = {'shuffle':True, 'worker_init_fn':partial(_init_fn, seed=args_speaker.optimization.seed)}
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        kwarg = {'sampler':train_sampler, 'worker_init_fn':partial(_init_fn, seed=args_speaker.optimization.seed)}
    train_dataloader = DataLoader(train_dataset, args_speaker.optimization.batch_size, num_workers=args.num_workers, pin_memory=True, **kwarg)
    test_dataloader = DataLoader(test_dataset, args_speaker.optimization.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainer = ClassifierTrainer(model, train_dataloader, optimizer, cost, batch_process, args.output_dir, 0, 
            test_dataloader, eval_hook=eval_hook, eval_every=args_speaker.optimization.N_eval_epoch, print_every=args_speaker.optimization.print_every, lr_scheduler=lr_scheduler)
    trainer.logger.info(args)
    trainer.run(args_speaker.optimization.N_epochs)


if __name__=="__main__":
    args = get_option()
    print(args)
    main(args)



