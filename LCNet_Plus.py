# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 11:48
# @Author  : xum
# @FileName: lcnet_plus.py

# adapted from https://github.com/ngnquan/PP-LCNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all___ = [
    'LCNetPlus_x0_25', 'LCNetPlus_x0_35', 'LCNetPlus_x0_5', 'LCNetPlus_x0_75',
    'LCNetPlus_x1_0', 'LCNetPlus_x1_5', 'LCNetPlus_x2_0', 'LCNetPlus_x2_5'
]


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0, ),
            nn.ReLU(),
            nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Hardswish(inplace=True),

                # SE
                SELayer(inp, inp),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardswish(inplace=True),

            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Hardswish(inplace=True),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.Hardswish(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class LCNetPlus(nn.Module):
    def __init__(self, scale=1.0, num_classes=1000, dropout_prob=0.2):
        super(LCNetPlus, self).__init__()
        self.cfgs = [
            # k,  c,  s, SE
            [3, 32, 1, 0],

            [3, 64, 2, 0],
            [3, 64, 1, 0],

            [3, 128, 2, 0],
            [3, 128, 1, 0],

            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],

            [5, 512, 2, 1],
            [5, 512, 1, 1],
        ]
        self.scale = scale

        input_channel = _make_divisible(16 * scale)
        layers = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.Hardswish(inplace=True)]

        block = DepSepConv
        for k, c, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(input_channel, 1280, 1, 1, 0)
        self.hwish = nn.Hardswish(inplace=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.hwish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def LCNetPlus_x0_25(**kwargs):
    """
    LCNetPlus_x0_25
    """
    return LCNetPlus(scale=0.25, **kwargs)


def LCNetPlus_x0_35(**kwargs):
    """
    LCNetPlus_x0_35
    """
    return LCNetPlus(scale=0.35, **kwargs)


def LCNetPlus_x0_5(**kwargs):
    """
    LCNetPlus_x0_5
    """
    return LCNetPlus(scale=0.5, **kwargs)


def LCNetPlus_x0_75(**kwargs):
    """
    LCNetPlus_x0_75
    """
    return LCNetPlus(scale=0.75, **kwargs)


def LCNetPlus_x1_0(**kwargs):
    """
    LCNetPlus_x1_0
    """
    return LCNetPlus(scale=1.0, **kwargs)


def LCNetPlus_x1_5(**kwargs):
    """
    LCNetPlus_x1_5
    """
    return LCNetPlus(scale=1.5, **kwargs)


def LCNetPlus_x2_0(**kwargs):
    """
    LCNetPlus_x2_0
    """
    return LCNetPlus(scale=2.0, **kwargs)


def LCNetPlus_x2_5(**kwargs):
    """
    LCNetPlus_x2_5
    """
    return LCNetPlus(scale=2.5, **kwargs)


if __name__ == "__main__":
    model = LCNetPlus_x0_25()
    sample = torch.rand([8, 3, 224, 224])
    out = model(sample)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
