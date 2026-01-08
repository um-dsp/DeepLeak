import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CPA(nn.Module):
    def __init__(self, in_dim, dim, stride=1, same=False, sc_x=True):
        super(CPA, self).__init__()
        self.dim = dim
        self.stride = stride
        self.same = same
        self.sc_x = sc_x
        self.cp_ffc = nn.Linear(in_dim, dim)
        self.bn = nn.BatchNorm2d(dim)

        if stride == 2 or same:
            if sc_x:
                self.cp_ffc_sc = nn.Linear(in_dim, dim)
                self.bn_sc = nn.BatchNorm2d(dim)
            if stride == 2:
                self.avgpool = nn.AvgPool2d(2)

    def forward(self, x, sc_x):
        _, c, w, h = x.shape
        out = rearrange(x, 'b c w h -> b w h c')
        out = self.cp_ffc(out)
        out = rearrange(out, 'b w h c -> b c w h')
        out = self.bn(out)

        if out.shape == sc_x.shape and self.sc_x:
            out += sc_x
        out = F.layer_norm(out, out.size()[1:])

        if self.stride == 2 or self.same:
            if self.sc_x:
                x = rearrange(sc_x, 'b c w h -> b w h c')
                x = self.cp_ffc_sc(x)
                x = rearrange(x, 'b w h c -> b c w h')
                x = self.bn_sc(x)
                out += x
            if not self.same:
                out = self.avgpool(out)
        return out

class SPA(nn.Module):
    def __init__(self, img, out=1):
        super(SPA, self).__init__()
        self.sp_ffc = nn.Sequential(nn.Linear(img**2, out**2))

    def forward(self, x):
        _, c, w, h = x.shape
        x = rearrange(x, 'b c w h -> b c (w h)')
        x = self.sp_ffc(x)
        l = x.shape[-1]
        return rearrange(x, 'b c (w h) -> b c w h', w=int(l**0.5), h=int(l**0.5))

class upa_block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2):
        super(upa_block, self).__init__()
        self.cat = cat
        self.stride = stride
        self.same = same
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, int(planes * w), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(planes * w)),
            nn.ReLU(),
            nn.Conv2d(int(planes * w), planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        ) if l != 1 else nn.Sequential(
            nn.Conv2d(in_planes, int(planes * w), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(planes * w)),
            nn.ReLU()
        )
        self.att = CPA(in_planes, planes, stride, same)

    def forward(self, x):
        out = self.cnn(x)
        out = self.att(x, out)
        return torch.cat([x, out], 1) if self.cat else out

class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums, num_classes=100, img=32):
        super(upanets, self).__init__()
        self.in_planes = filter_nums
        w = 2

        self.root = nn.Sequential(
            nn.Conv2d(3, int(filter_nums * w), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(filter_nums * w)),
            nn.ReLU(),
            nn.Conv2d(int(filter_nums * w), filter_nums, 3, padding=1, bias=False),
            nn.BatchNorm2d(filter_nums),
            nn.ReLU()
        )
        self.emb = CPA(3, filter_nums, same=True)

        self.layer1 = self._make_layer(block, filter_nums * 1, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, filter_nums * 2, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, filter_nums * 4, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, filter_nums * 8, num_blocks[3], 2)

        self.spa0 = SPA(img)
        self.spa1 = SPA(img)
        self.spa2 = SPA(int(img * 0.5))
        self.spa3 = SPA(int(img * 0.25))
        self.spa4 = SPA(int(img * 0.125))

        self.linear = nn.Linear(filter_nums * 31, num_classes)
        self.bn = nn.BatchNorm1d(filter_nums * 31)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        planes = planes // num_blocks
        for i in range(num_blocks):
            if i == 0 and stride == 1:
                layers.append(block(self.in_planes, self.in_planes, stride, same=True))
            elif stride == 1:
                layers.append(block(self.in_planes, planes, stride, cat=True))
                self.in_planes += planes
            else:
                layers.append(block(self.in_planes, self.in_planes, stride))
                self.in_planes = self.in_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out01 = self.root(x)
        out0 = self.emb(x, out01)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out0 = F.avg_pool2d(out0, out0.size()[2:]) + self.spa0(out0)
        out1 = F.avg_pool2d(out1, out1.size()[2:]) + self.spa1(out1)
        out2 = F.avg_pool2d(out2, out2.size()[2:]) + self.spa2(out2)
        out3 = F.avg_pool2d(out3, out3.size()[2:]) + self.spa3(out3)
        out4 = F.avg_pool2d(out4, out4.size()[2:]) + self.spa4(out4)

        out = torch.cat([F.layer_norm(x, x.size()[1:]) for x in [out4, out3, out2, out1, out0]], 1)
        return self.linear(self.bn(out.view(out.size(0), -1)))

def UPANets(f, c=100, block=1, img=32):
    return upanets(upa_block, [int(4 * block)] * 4, f, num_classes=c, img=img)
