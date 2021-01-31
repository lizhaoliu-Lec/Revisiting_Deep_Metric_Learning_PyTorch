"""
The network architectures and weights are adapted and used
from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import pretrainedmodels as ptm
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, opt, return_embed_dict=False):
        super(Network, self).__init__()

        self.pars = opt
        self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        if '_he' in opt.arch:
            nn.init.kaiming_normal_(self.model.last_linear.weight, mode='fan_out')
            nn.init.constant_(self.model.last_linear.bias, 0)

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.return_embed_dict = return_embed_dict

        self.pool_base = nn.AdaptiveAvgPool2d(1)
        self.pool_aux = nn.AdaptiveMaxPool2d(1) if 'double' in opt.arch else None

        self.name = opt.arch

        self.out_adjust = None

    def forward(self, x, warmup=False, **kwargs):
        x = self.model.features(x)
        x_pooled = self.pool_base(x)
        if self.pool_aux is not None:
            x_pooled += self.pool_aux(x)
        if warmup:
            x_pooled, x = x_pooled.detach(), x.detach()
        embedding = self.model.last_linear(x_pooled.view(len(x), -1))
        if 'normalize' in self.name:
            embedding = F.normalize(embedding, dim=-1)
        if self.out_adjust and not self.training:
            embedding = self.out_adjust(embedding)
        return embedding, (x_pooled, x)

    def functional_forward(self, x):
        pass
