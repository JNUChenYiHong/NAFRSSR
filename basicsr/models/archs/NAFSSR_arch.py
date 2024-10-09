'''
@misc{chen2024nafrssr,
      title={NAFRSSR: a Lightweight Recursive Network for Efficient Stereo Image Super-Resolution}, 
      author={Yihong Chen and Zhen Fan and Shuai Dong and Zhiwei Chen and Wenjie Li and Minghui Qin and Min Zeng and Xubing Lu and Guofu Zhou and Xingsen Gao and Jun-Ming Liu},
      year={2024},
      eprint={2405.08423},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()

        # kernel_1和kernel2为： 边缘检测
        # kernel_1 = [[-1, -1, -1],
        #             [-1, 8, -1],
        #             [-1, -1, -1]]

        kernel_2 = [[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]]

        kernel = torch.Tensor(kernel_2).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=True)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]

        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)

        x4 = F.conv2d(x4.unsqueeze(1), self.weight, padding=1)
        x5 = F.conv2d(x5.unsqueeze(1), self.weight, padding=1)
        x6 = F.conv2d(x6.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        return x


# my module
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=c // 4, bias=True)

        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
                               groups=1,
                               bias=True)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # block2
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
        #                        groups=c//4, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        # self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.l_proj1(self.norm_l(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        # Q_l = self.norm_l(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        # Q_r_T = self.norm_r(x_r).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = x_l.permute(0, 2, 3, 1)  # B, H, W, c
        V_r = x_r.permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r


class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x + factor * (new_x - x) for x, new_x in zip(feats, new_feats)])
        return new_feats


class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''

    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats


class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''

    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0.,
                 fusion_from=-1, fusion_to=-1, dual=False):
        super().__init__()
        self.dual = dual  # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate,
                NAFBlockSR(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        self.blur = GaussianBlur()
        self.weight = nn.Parameter(torch.zeros((1, 6, 1, 1)), requires_grad=True)

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bicubic')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp,)
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)

        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + self.blur(out) * self.weight
        out = out + inp_hr
        return out


class NAFSSR(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    num_blks = 34  # NAFRSSR-S
    width = 80
    droppath = 0.1
    train_size = (1, 6, 30, 90)

    net = NAFSSR(up_scale=4, train_size=train_size, fast_imp=True, width=width, num_blks=num_blks,
                 drop_path_rate=droppath)

    inp_shape = (6, 64, 64)

    from ptflops import get_model_complexity_info

    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))