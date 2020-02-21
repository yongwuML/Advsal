from train.model import resnext_101_32x4d_
import torch
from torch import nn
from train.admin.environment import env_settings


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        resnext101_32_path = env_settings().resnext101_32_dir
        net = resnext_101_32x4d_.resnext_101_32x4d
        net.load_state_dict(torch.load(resnext101_32_path))

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:4])
        self.layer1 = net[4]
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4