import torch
import torch.nn as nn
from core.layers.MAUCell import MAUCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.in_channel
        self.out_frame_channel = configs.patch_size * configs.patch_size * configs.out_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []

        width = configs.img_width // configs.patch_size // configs.sr_size  # 512 // 4 // 4 = 32
        height = configs.img_height // configs.patch_size // configs.sr_size # 512 // 4 // 4 = 32
        # print(width)

        for i in range(num_layers):
            # MAUCellのインスタンス化
            cell_list.append(MAUCell(num_hidden[i-1], num_hidden[i], height, width, configs.filter_size, configs.stride, self.tau, self.cell_mode))
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size)) # 2
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1), module=nn.Conv2d(in_channels=self.frame_channel, out_channels=self.num_hidden[0], stride=1, padding=0, kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1), module=nn.LeakyReLU(0.2))
        encoders.append(encoder)

        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i), module=nn.Conv2d(in_channels=self.num_hidden[0], out_channels=self.num_hidden[0], stride=(2, 2), padding=(1, 1), kernel_size=(3, 3) ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i), module=nn.LeakyReLU(0.2))
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []
        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], stride=(2, 2), padding=(1, 1), kernel_size=(3, 3), output_padding=(1, 1)))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1], out_channels=self.num_hidden[-1], stride=(2, 2), padding=(1, 1), kernel_size=(3, 3), output_padding=(1, 1) ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        #self.srcnn = nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        self.srcnn = nn.Conv2d(self.num_hidden[-1], self.out_frame_channel, kernel_size=1, stride=1, padding=0)
        #self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        #self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)


    def forward(self, frames, mask_true):
        """
        Forward pass of the MAU model.

        Args:
            frames (torch.Tensor): Input frames.
            mask_true (torch.Tensor): Ground truth masks (but NOT Ground truth frames)

        Returns:
            next_frames (torch.Tensor): Predicted frames.
        """
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size # 512 // 4 = 128
        width = frames.shape[4] // self.configs.sr_size # 512 // 4 = 128
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s.append(torch.zeros([batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length: # 0~11 (Input frames)
                net = frames[:, t]
            else:                             # 12~23 (Predicted frames)
                time_diff = t - self.configs.input_length
                #net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                channel = int(frames.shape[2] // (self.configs.in_channel / self.configs.out_channel)) # (128*3) // (3 / 1) = 128
                """
                print(mask_true.shape, mask_true[:, time_diff, :channel].shape)
                print(frames.shape, frames[:, t, :channel].shape)
                print(x_gen.shape)
                print(channel)
                """
                net = mask_true[:, time_diff, :channel] * frames[:, t, :channel] + (1 - mask_true[:, time_diff, :channel]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            
            # エンコーダーをスタック
            # tに合わせてエンコーダーの形を変える必要がある
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)

            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t.append(zeros)

            S_t = frames_feature
            # num_layersで指定した階層分MAUをスタック
            for k in range(self.num_layers):
                T_att = torch.stack(T_pre[k][-self.tau:], dim=0)
                S_att = torch.stack(S_pre[k][-self.tau:], dim=0)
                S_pre[k].append(S_t)
                T_t[k], S_t = self.cell_list[k](T_t[k], S_t, T_att, S_att) #MAUのforward
                T_pre[k].append(T_t[k])
            out = S_t

            # デコーダーをスタック
            frames_feature_decoded = []
            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            x_gen = self.srcnn(out)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        return next_frames