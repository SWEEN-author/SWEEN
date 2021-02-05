import torch
from model.resnet import resnet20, resnet26, resnet32, resnet110
from model.densenet import densenet_BC_cifar
from model.vgg import vgg16, vgg19
from model.alexnet import alexnet
from model.lenet import lenet


ARCHITECTURES = ['lenet', 'alexnet', 'resnet20', 'resnet26', 'resnet32',
                 'resnet110', 'densenet', 'vgg16', 'vgg19']

def get_architecture(arch: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)
    :param arch: the architecture - should be in the ARCHITECTURES list above
    :return: a Pytorch module
    """
    if arch == 'lenet':
        model = lenet()
    elif arch == 'alexnet':
        model = alexnet()
    elif arch == 'resnet20':
        model = resnet20()
    elif arch == 'resnt26':
        model = resnet26()
    elif arch == 'resnet32':
        model = resnet32()
    elif arch == 'resnet110':
        model = resnet110()
    elif arch == 'densenet':
        model = densenet_BC_cifar(depth=100, k=12)
    elif arch == 'vgg16':
        model = vgg16()
    elif arch == 'vgg19':
        model = vgg19()
    else:
        raise ValueError('arch not in ARCHITECTURES')
    return model