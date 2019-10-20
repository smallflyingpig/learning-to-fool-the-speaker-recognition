import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np 
import logging
import tqdm
import pickle
from tensorboardX import SummaryWriter
from optparse import OptionParser
from dataset import TIMIT_speech, TIMIT_speaker
from model import SincClassifier
from utils import read_conf, get_dict_from_args
from trainer import ClassifierTrainer, save_checkpoint, load_checkpoint
import multiprocessing
multiprocessing.set_start_method('spawn', True)


def get_optimizer(opt_type, params, lr, **kwargs):
    opt_dict = {'sgd':optim.SGD, 'rmsprop':optim.RMSprop, 'adam':optim.Adam}
    return opt_dict[opt_type](params, lr=lr, **kwargs)

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
        pred_batch = torch.nn.functional.softmax(speaker_model(torch.stack(batch_data)), dim=1)
        pred_all.append(pred_batch)
    [val,best_class]=torch.max(torch.sum(torch.cat(pred_all, dim=0),dim=0),0)
    return best_class.detach().cpu().item()

import soundfile as sf
@torch.no_grad()
def test_wav(model, filename_list, data_folder, label_dict):
    model.eval()
    bar = tqdm.tqdm(filename_list)
    err_rate = 0
    for idx, filename in enumerate(bar):
        real_data, fs = sf.read(os.path.join(data_folder, filename))
        label = label_dict['/'.join(filename.split('/')[-4:])]
        # label = label_dict[filename.split('/')[-2]]
        pred = sentence_test(model, torch.from_numpy(real_data).float().cuda().unsqueeze(0))
        err_rate += (pred != label)
        bar.set_description("label/pred:{}/{}".format(label, pred))
    bar.close()
    err_rate = err_rate/float(len(filename_list))
    print("err rate:", err_rate)

@torch.no_grad()
def evaluate(model, test_dataset, cost=None, ):
    test_dataloader = DataLoader(test_dataset, 128, shuffle=False, num_workers=8, pin_memory=True)
    print("dataset len:", len(test_dataset))
    model.eval()
    bar = tqdm.tqdm(test_dataloader)
    loss_total = 0
    err_total = 0
    for idx, data in enumerate(bar):
        wav_data, label = data
        wav_data, label = wav_data.float().cuda().squeeze(0), label.long().cuda().squeeze(0)
        loss_func = cost
        # print(wav_data.shape, label.shape)
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float()).detach().item()
        if loss_func is not None:
            loss = loss_func(pout, label).detach().item()
        else:
            loss = -1
        loss_total += loss * len(label)
        err_total += err * len(label)
        bar.set_description("batch:(loss:{:5.3f},err:{:5.3f},batch size:{})".format(loss, err, len(label)))
    loss_total /= len(test_dataset)
    err_total /= len(test_dataset)
    print("eval result: loss: {:5.3f}, err:{:5.3f}".format(loss_total, err_total))


def batch_process_speech(model:SincClassifier, data, train_mode=True, **kwargs)->dict:
    wav_data, label = data
    wav_data, label = wav_data.float().cuda(), label.long().cuda()
    if train_mode:
        model.train()
        optimizers, loss_func = kwargs.get('optimizer'), kwargs.get('loss_func')
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float())
        loss = loss_func(pout, label)
        model.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        loss, err = loss.detach().item(), err.detach().item()
        rtn = {
             "output":"loss:{:6.3f}, err:{:5.3f}, lr(e-3):[{:6.3f}, {:6.3f}, {:6.3f}]".format(
                 loss, err, optimizers[0].param_groups[0]['lr']*1000, 
                 optimizers[1].param_groups[0]['lr']*1000, optimizers[2].param_groups[0]['lr']*1000),
            "vars":{"loss":loss, "err":err},
            "count":{"loss":len(label), "err":len(label)}
        }
    else: #eval
        model.eval()
        wav_data, label = wav_data.squeeze(0), label.squeeze(0)
        loss_func = kwargs.get("loss_func")
        # print(wav_data.shape, label.shape)
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float())

        loss = loss_func(pout, label)
        loss, err = loss.detach().item(), err.detach().item()
        rtn = {
            "output":"loss:{:6.3f}, err:{:5.3f}".format(loss, err),
            "vars":{"loss":loss, "err":err},
            "count":{"loss":len(label), "err":len(label)}
        }
    return rtn


def batch_process_speaker(model:SincClassifier, data, train_mode=True, **kwargs)->dict:
    wav_data, label = data
    wav_data, label = wav_data.float().cuda(), label.long().cuda()
    if train_mode:
        model.train()
        optimizers, loss_func = kwargs.get('optimizer'), kwargs.get('loss_func')
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float())
        loss = loss_func(pout, label)
        model.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        loss, err = loss.detach().item(), err.detach().item()
        rtn = {
             "output":"loss:{:6.3f}, err:{:5.3f}, lr(e-3):[{:6.3f}, {:6.3f}, {:6.3f}]".format(
                 loss, err, optimizers[0].param_groups[0]['lr']*1000, 
                 optimizers[1].param_groups[0]['lr']*1000, optimizers[2].param_groups[0]['lr']*1000),
            "vars":{"loss":loss, "err":err},
            "count":{"loss":len(label), "err":len(label)}
        }
    else: #eval
        model.eval()
        wav_data, label = wav_data.squeeze(0), label.squeeze(0)
        loss_func = kwargs.get("loss_func")
        # print(wav_data.shape, label.shape)
        pout = model.forward(wav_data)
        pred = torch.max(pout, dim=1)[1]
        err = torch.mean((pred != label).float())

        loss = loss_func(pout, label)
        loss, err = loss.detach().item(), err.detach().item()
        rtn = {
            "output":"loss:{:6.3f}, err:{:5.3f}".format(loss, err),
            "vars":{"loss":loss, "err":err},
            "count":{"loss":len(label), "err":len(label)}
        }
    return rtn

class EvalHook(object):
    def __init__(self):
        self.best_accu = 0
    
    def __call__(self, model:nn.Module, epoch_idx, output_dir, 
        eval_rtn:dict, test_rtn:dict, logger:logging.Logger, writer:SummaryWriter):
        # save model
        is_best = 1-eval_rtn.get('err', 0) > self.best_accu
        self.best_accu = 1-eval_rtn.get('err', 0) if is_best else self.best_accu
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


def get_option():
    parser=OptionParser()
    parser.add_option("--type", choices=['speech', 'speaker'], default="speech") # Mandatory
    parser.add_option("--eval", action='store_true', default=False, help="eval the model")
    parser.add_option("--test", action='store_true', default=False, help="sentence test for the speaker model")
    parser.add_option("--pt_file", type=str, default='', help="path for pretrained file")
    parser.add_option("--data_root", type=str, default='./data/TIMIT/TIMIT_lower', help="path for data")
    parser.add_option("--output_dir", type=str, default="./output/sincnet")
    parser.add_option("--dataset", choices=['timit', 'libri'], default='timit', help="the dataset name")
    parser.add_option("--num_workers", type=int, default=8, help="num workers for dataloader")
    (args,_)=parser.parse_args()
    return args


datasets = {"timit":{"speech":TIMIT_speech, "speaker":TIMIT_speaker}}
batch_processes = {"speech":batch_process_speech, "speaker":batch_process_speaker}
def main(args):
    args.cfg = "./config/{}_{}.cfg".format(args.dataset, args.type)
    args = read_conf(args.cfg, args)
    torch.manual_seed(args.optimization.seed)
    np.random.seed(args.optimization.seed)
    train_dataset = datasets[args.dataset][args.type](args.data_root, train=True)
    test_dataset = datasets[args.dataset][args.type](args.data_root, train=False)
    
    cost = nn.CrossEntropyLoss()

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
    if args.pt_file!='':
        print("load model from:", args.pt_file)
        checkpoint_load = torch.load(args.pt_file)
        ext = os.path.splitext(args.pt_file)[1]
        if ext == '.pkl':
            model.load_raw_state_dict(checkpoint_load)
        elif ext == '.pickle':
            model.load_state_dict(checkpoint_load)
        elif ext == '.pth':
            load_checkpoint(model, args.pt_file)
        else:
            raise NotImplementedError
    model = model.cuda()
    if args.eval:
        print('only eval the model')
        evaluate(model, test_dataset, cost)
        return
    if args.test and args.type=='speaker':
        print("only test the model")
        filename_list = open("./data/TIMIT/speaker/test.scp", 'r').readlines()
        filename_list = [_f.strip() for _f in filename_list]
        # with open(os.path.join(args.data_root, "processed", "speaker_id.pickle"), "rb") as fp:
        #     label_dict = pickle.load(fp)
        label_dict = np.load(os.path.join(args.data_root, "processed", "TIMIT_labels.npy")).item()
        test_wav(model, filename_list, args.data_root, label_dict)
        return 

    print("train the model")
    batch_process = batch_processes[args.type]
    eval_hook = EvalHook()
    optimizer = [get_optimizer(args.cnn.arch_opt, model.CNN_net.parameters(), args.cnn.arch_lr),
            get_optimizer(args.dnn.arch_opt, model.DNN_net.parameters(), args.dnn.arch_lr),
            get_optimizer(args.classifier.arch_opt, model.Classifier.parameters(), args.classifier.arch_lr),] 
    lr_scheduler = [optim.lr_scheduler.StepLR(optimizer[0], args.cnn.lr_decay_step, args.cnn.lr_decay_factor),
            optim.lr_scheduler.StepLR(optimizer[1], args.dnn.lr_decay_step, args.dnn.lr_decay_factor),
            optim.lr_scheduler.StepLR(optimizer[2], args.classifier.lr_decay_step, args.classifier.lr_decay_factor)]
    train_dataloader = DataLoader(train_dataset, args.optimization.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, args.optimization.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    trainer = ClassifierTrainer(model, train_dataloader, optimizer, cost, batch_process, args.output_dir, 0, 
            test_dataloader, eval_hook=eval_hook, eval_every=args.optimization.N_eval_epoch, print_every=args.optimization.print_every, lr_scheduler=lr_scheduler)
    trainer.run(args.optimization.N_epochs)


if __name__=="__main__":
    args = get_option()
    print(args)
    main(args)



