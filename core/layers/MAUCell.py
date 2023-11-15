import torch
import torch.nn as nn
import math


class MAUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError

        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,),
            nn.LayerNorm([3 * num_hidden, height, width])
        )

        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,),
            nn.LayerNorm([num_hidden, height, width])
        )

        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_prev, S_lower, T_att, S_att):
        # INPUTS:
        # T_prev: T_(k)_(t-1) in paper
        # S_lower: S_(k-1)_(t) in paper
        # T_att: T_(k)_(t-j) in paper
        # S_att: S_(k-1)_(t) in paper
        
        # -------------- Attention Module --------------
        S_lower = self.conv_s_next(S_lower)

        # (3) in paper
        weights_list = []
        for i in range(self.tau):
            weights_list.append((S_att[i] * S_lower).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)

        # (4) in paper
        T_att = (T_att * weights_list).sum(dim=0)

        # (5) in paper
        U_f = torch.sigmoid(self.conv_t_next(T_prev))
        T_AMI = T_prev * U_f + (1 - U_f) * T_att 

        # -------------- Fusion Module --------------
        
        # W_ttやW_ssなどをかける計算
        # ここでの添字は分割の順番を表すものであり、tは時系列番号のtに関係がない
        T_AMI_wu, T_AMI_wt, T_AMI_ws = torch.split(self.conv_t(T_AMI),   self.num_hidden, dim=1)
        S_wu, S_wt, S_ws             = torch.split(self.conv_s(S_lower), self.num_hidden, dim=1)
        
        #(6) in paper     
        U_t = torch.sigmoid(T_AMI_wu)
        U_s = torch.sigmoid(S_wu)

        # (7) in paper
        T_out = U_t * T_AMI_wt + (1 - U_t) * S_wt
        S_out = U_s * S_ws + (1 - U_s) * T_AMI_ws
        if self.cell_mode == 'residual':
            S_out = S_out + S_lower

        return T_out, S_out
