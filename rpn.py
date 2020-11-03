import torch
import torch.nn as nn


class RPN(nn.Module):
    def __init__(self, RPN):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d()