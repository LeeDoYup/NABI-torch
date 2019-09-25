from __future__ import print_function
import torch
import torch.nn as nn


class SpatialNL(nn.Module):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
       revised to 1d
    """
    def __init__(self, inplanes, planes, use_scale=False):
        self.use_scale = use_scale

        super(SpatialNL, self).__init__()
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, d = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, d)

        x = self.z(x)
        x = self.bn(x) + residual

        return x
    
    
if __name__ == '__main__':
    NLNN = SpatialNL(10, 20)
    print(NLNN)