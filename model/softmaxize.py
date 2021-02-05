import torch
import torch.nn as nn
import torch.nn.functional as F


class Softmaxize(nn.Module):
    def __init__(self, net):
        super(Softmaxize, self).__init__()
        self.net = net
    def forward(self, x):
        out = self.net(x)
        out = F.softmax(out, dim=1)
        return(out)