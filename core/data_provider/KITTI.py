from __future__ import print_function, division

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import codecs
from core.utils import preprocess



class Norm(object):
    def __init__(self, max=255):
        self.max = max

    def __call__(self, sample):
        video_x = sample
        new_video_x = video_x / self.max
        return new_video_x


class ToTensor(object):
    def __call__(self, sample):
        video_x = sample
        video_x = video_x.transpose((0, 3, 1, 2))
        video_x = np.array(video_x)
        return torch.from_numpy(video_x).float()


class KITTI(Dataset):

    def __init__(self, configs, data_train_path, data_test_path, mode, transform=None):
        self.transform = transform
        self.mode = mode
        self.configs = configs
        self.patch_size = configs.patch_size
        self.img_width = configs.img_width
        self.img_height = configs.img_height
        self.in_channel = configs.in_channel
        if self.mode == 'train':
            print('Loading train dataset')
            self.path = data_train_path
            with codecs.open(self.path) as f:
                self.file_list = f.readlines()
            print('Loading train dataset finished, with size:', len(self.file_list))
        else:
            print('Loading test dataset')
            self.path = data_test_path
            with codecs.open(self.path) as f:
                self.file_list = f.readlines()
            print('Loading test dataset finished, with size:', len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item_ifo_list = self.file_list[idx].split(',')
        begin = int(item_ifo_list[1])
        end = begin + self.configs.total_length
        data_slice = np.ndarray(shape=(end - begin, self.img_height, self.img_width, self.in_channel), dtype=np.uint8)
        key_words_list = item_ifo_list[0].split('/')
        if key_words_list[2] == 'KITTI':
            for i in range(end - begin):
                file_index = i + begin
                base_str = ''
                for _ in range(10 - len(str(file_index))):
                    base_str += '0'
                base_str += str(file_index)
                file_name = base_str + '.png'
                image = cv2.imread(str(item_ifo_list[0]) + file_name)
                data_slice[i, :] = image
        else:
            for i in range(end - begin):
                file_index = i + begin
                file_name = str(file_index) + '.png'
                image = cv2.imread(str(item_ifo_list[0]) + file_name)
                data_slice[i, :] = image

        video_x = preprocess.reshape_patch(data_slice, self.patch_size)
        sample = video_x

        if self.transform:
            sample = self.transform(sample)

        return sample

