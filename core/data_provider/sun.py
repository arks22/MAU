from __future__ import print_function, division
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from core.utils import preprocess


class ToTensor(object):
    def __call__(self, sample):
        video_x = sample
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        return torch.from_numpy(video_x).float()


class sun(Dataset):
    def __init__(self, configs, path, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.configs = configs
        self.patch_size = configs.patch_size
        self.img_width = configs.img_width
        self.img_height = configs.img_height
        self.in_channel = configs.in_channel
        self.path = path

        data_size = os.path.getsize(self.path)
        
        if data_size > 20 * 1024 * 1024 * 1024:
            # 40gbよりデカい場合,メモリマップで読み込む
            self.data = np.load(self.path, mmap_mode='r')
            print('Loading with memory mapping', mode, 'dataset finished, with size:', self.data.shape[1])
        else:
            self.data = np.load(self.path)
            print('Loading', mode, 'dataset finished, with size:', self.data.shape[1])


    def __len__(self):
        return self.data.shape[1]


    def __getitem__(self, idx):
        # DataLoaderオブジェクトを作成する際にこのsunクラスのインスタンスを渡すと、
        # データローダーがバッチを生成する際に自動的に__getitem__が呼び出される
        sample = self.data[:, idx, :]

        if self.transform:
            sample = preprocess.reshape_patch(sample, self.patch_size)
            sample = self.transform(sample)
        return sample
