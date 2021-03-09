"""
The network architectures and weights are adapted and used from the great
https://github.com/Cadene/pretrained-models.pytorch.
"""
import pretrainedmodels as ptm
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000,
                                              pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        if 'bn_norm' in opt.arch:
            print("Using BN Norm after the final linear!!!")
            self.bn = nn.BatchNorm1d(num_features=opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.out_adjust = None

    def forward(self, x, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        x_before_pooled = x
        x = self.model.avgpool(x)
        x_pooled = x = x.view(x.size(0), -1)

        x = self.model.last_linear(x)

        if 'bn_norm' in self.pars.arch:
            x = self.bn(x)

        if 'normalize' in self.pars.arch:
            x = nn.functional.normalize(x, dim=-1)
        if self.out_adjust and not self.train:
            x = self.out_adjust(x)

        return x, (x_pooled, x_before_pooled)
