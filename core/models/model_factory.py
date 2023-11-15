import os
import glob
import torch
import torch.nn as nn
from torch.optim import Adam
from core.models import RNN
import torch.optim.lr_scheduler as lr_scheduler


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.patch_height = configs.img_height // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_channel = configs.in_channel * (configs.patch_size ** 2)
        self.num_layers = configs.num_layers

        networks_map = {
            'mau': RNN.RNN,
        }
        num_hidden = []

        # MAUセルの深さに合わせてCNNのチャンネル数を設定
        num_hidden = [configs.num_hidden for i in range(configs.num_layers)]

        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device) #RNNのインスタンス化
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=configs.lr_decay)
        self.MSE_criterion = nn.MSELoss()
        self.L1_loss = nn.L1Loss()


    def save(self, timestamp, itr):
        stats = {'net_param': self.network.state_dict()}

        checkpoint_path = os.path.join(self.configs.save_dir,timestamp, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("\nsave predictive model to %s" % checkpoint_path)


    def load(self, pm_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])


    def train(self, frames, mask, itr):
        self.network.train()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)

        # RNNのforward (インスタンスにデータを渡すことで間接的にfowardが呼ばれる)
        next_frames = self.network(frames_tensor, mask_tensor) 
        ground_truth = frames_tensor

        self.optimizer.zero_grad()
        loss_l1 = self.L1_loss(next_frames, ground_truth[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames, ground_truth[:, 1:])
        loss_gen = loss_l2
        loss_gen.backward()
        self.optimizer.step()

        if itr >= self.configs.sampling_stop_iter and itr % self.configs.delay_interval == 0:
            self.scheduler.step()
            print('Lr decay to {:.8f}'.format(self.optimizer.param_groups[0]['lr']))

        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy()

    def test(self, frames, mask):
        self.network.eval()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()
