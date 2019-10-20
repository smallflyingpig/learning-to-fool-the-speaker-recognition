import torch
import torchvision
from torch.utils.data import Dataset
import os
import os.path as osp
import numpy as np 
import torchaudio
import scipy 
import pickle
import csv
import pandas as pd
import soundfile as sf


def read_list(filename):
    with open(filename, "r") as fp:
        data = fp.readlines()
        data = [_l.strip() for _l in data]
    return data

def load_pickle(filenmae, encoding='utf8'):
    with open(filenmae, mode="rb") as fp:
        data = pickle.load(fp, encoding=encoding)
    return data

class TIMIT_base(Dataset):
    def __init__(self):
        super(TIMIT_base, self).__init__()

    @staticmethod
    def preprocess(wav_data):
        norm_factor = np.abs(wav_data).max()
        wav_data = wav_data/norm_factor
        return wav_data, norm_factor

    def load_frame(self, wav_filename, offset, f_wlen):
        wav_data, fs = sf.read(wav_filename)
        # assert offset+f_wlen<=len(wav_data)
        wav_data, norm_factor =  self.preprocess(wav_data)# normlize
        offset = min(max(offset, 0), len(wav_data)-f_wlen)
        frame = wav_data[offset:offset+f_wlen]
        return frame, norm_factor

# with data augmentation now
class TIMIT_speaker(TIMIT_base):
    def __init__(self, data_root, train=True, fs=16000, wlen=200, wshift=10, phoneme=False, norm_factor=False, augment=True):
        super(TIMIT_speaker, self).__init__()
        data_root_processed = osp.join(data_root, "processed")
        self.data_root = data_root
        self.fs, self.wlen = fs, wlen
        self.f_wlen = int(fs*wlen/1000)
        self.f_wshift = int(fs*wshift/1000)
        self.split = "train" if train else "test"
        self.phoneme = phoneme
        self.norm_factor = norm_factor
        self.augment = augment
        # read csv and speaker id file
        table_file = osp.join(data_root_processed, "speaker_{}.csv".format(self.split))
        print("load table file from: ", table_file)
        self.data = pd.read_csv(table_file)
        print("table keys: ", self.data.keys())
        speaker_id_file = osp.join(data_root_processed, "speaker_id.pickle")
        print("load speaker id from: ", speaker_id_file)
        self.speaker_id = load_pickle(speaker_id_file)
        print("phone keys len: ", len(self.speaker_id))
        self.timit_labels = np.load(os.path.join(data_root_processed, "TIMIT_labels.npy")).item()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index, :]
        if self.augment and self.split == 'train':
            offset = row['offset'] + np.random.randint(-self.f_wshift, self.f_wshift)
        else:
            offset = row['offset']

        frame, norm_factor = self.load_frame(osp.join(self.data_root, row['filename']), offset, self.f_wlen)
        # if self.augment and self.split == 'train':
        #    frame *= np.random.uniform(1-0.2, 1+0.2)
        speaker_id, phoneme = row['speaker_id'], row['phoneme']
        speaker_id = self.timit_labels[row['filename']]
        rtn = (frame, speaker_id)
        if self.phoneme:
            rtn += (phoneme,)
        if self.norm_factor:
            rtn += (norm_factor,)
        return rtn

