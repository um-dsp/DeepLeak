import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t=6, downsample=False):
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.shortcut = (not downsample) and (input_channel == output_channel)

        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        c = t * input_channel

        self.conv1 = nn.Conv2d(input_channel, c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, stride=self.stride, padding=1, groups=c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv3 = nn.Conv2d(c, output_channel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.shortcut:
            out += x
        return out

class MobileNetV2(nn.Module):
    def __init__(self, output_size, alpha=1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        self.conv0 = nn.Conv2d(3, int(32 * alpha), 3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(int(32 * alpha))

        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t=1, downsample=False),
            BaseBlock(16, 24, downsample=False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample=False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample=True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample=False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample=True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample=False)
        )

        self.conv1 = nn.Conv2d(int(320 * alpha), 1280, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.relu6(self.bn0(self.conv0(x)))
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.fc(x)

class MobileNetV2BCE(MobileNetV2):
    def __init__(self, output_size, alpha=1, dropout_prob=0.5):
        super().__init__(output_size, alpha)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(1280, 1)

    def forward(self, x):
        x = F.relu6(self.bn0(self.conv0(x)))
        x = self.bottlenecks(x)
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))
