import architectures.bninception
import architectures.googlenet
import architectures.resnet50


def select(arch, opt):
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    elif 'googlenet' in arch:
        return googlenet.Network(opt)
    elif 'bninception' in arch:
        return bninception.Network(opt)
    else:
        raise ValueError('get unrecognized arch')
