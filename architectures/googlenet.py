"""
The network architectures and weights are adapted and used from the great
https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.model = mod.googlenet(pretrained=True)

        self.model.last_linear = nn.Linear(self.model.fc.in_features, opt.embed_dim)
        self.model.fc = self.model.last_linear

        self.name = opt.arch

    def forward(self, x):
        x = self.model(x)
        if 'normalize' not in self.pars.arch:
            return x
        return F.normalize(x, dim=-1)
