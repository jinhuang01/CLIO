'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mobilenetv2 import mobilenet_v2
from .utils import process_state_dict
import copy

import numpy as np

__all__ = ['CLIOMobileNetV2']

class CLIOMobileNetV2(nn.Module):

    def __init__(self, split_point=5, widths=[1,2,4,8,16,32], use_random_connect=True, default_width=32, num_classes=20, pretrained=True, reload_path=None):
        super(CLIOMobileNetV2, self).__init__()

        self.widths = widths
        self.default_width = default_width
        self.use_random_connect = use_random_connect

        net = mobilenet_v2(pretrained=pretrained)

        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(num_ftrs, num_classes)

        if reload_path:
            checkpoint = torch.load(reload_path)
            net.load_state_dict(process_state_dict(checkpoint['state_dict']))
        
        self.shared_layers = net.features[0:split_point+1]
        self.heads = nn.ModuleDict()
        self.linears = nn.ModuleDict()
        in_channels = net.features[split_point].conv[2].in_channels
        for w in self.widths:
            base_net = copy.deepcopy(net.features)
            base_net[split_point].conv[2] = nn.Conv2d(in_channels, w, kernel_size=1, stride=1, padding=0, bias=False)
            torch.nn.init.xavier_uniform_(base_net[split_point].conv[2].weight)
            
            base_net[split_point].conv[3] = nn.BatchNorm2d(w)
            base_net[split_point].use_res_connect = False

            out_channels = base_net[split_point + 1].conv[0][0].out_channels
            base_net[split_point + 1].conv[0][0] = torch.nn.Conv2d(w, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            torch.nn.init.xavier_uniform_(base_net[split_point + 1].conv[0][0].weight)
            
            base_net[split_point + 1].use_res_connect = True
            base_net[split_point + 1].shortcut = nn.Sequential(
                nn.Conv2d(
                    w, 
                    base_net[split_point + 1].conv[2].out_channels, 
                    kernel_size=1, 
                    stride=base_net[split_point + 1].conv[1][0].stride, 
                    padding=0, 
                    bias=False),
                nn.BatchNorm2d(base_net[split_point + 1].conv[2].out_channels),
            )
            base_net = base_net[split_point + 1:]
            self.heads[str(w)] = base_net

            self.linears[str(w)] = copy.deepcopy(net.classifier)

    def forward(self, x, width=None):
        if width is None:
            width = self.default_width
            if self.use_random_connect:
                width = np.random.choice(self.widths)
        
        out = self.shared_layers(x)
        out = out[:, 0:width, :, :]
        out = self.heads[str(width)](out)
        out = out.mean([2, 3])
        out = self.linears[str(width)](out)
        return out


def test():
    net = CLIOMobileNetV2()
    print(net)
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

# test()
