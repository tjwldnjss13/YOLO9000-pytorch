import torch
import torch.nn as nn


class RPN(nn.Module):
    def __init__(self, n_filter, n_anchor_box=5):
        super(RPN, self).__init__()
        self.n_anchor_box = n_anchor_box
        self.reg_conv1 = nn.Conv2d(n_filter, 512, 1, 1, 0)
        self.reg_conv2 = nn.Conv2d(512, 5 * self.n_anchor_box, 1, 1, 0)
        self.cls_conv1 = nn.Conv2d(n_filter, 256, 1, 1, 0)
        self.cls_conv2 = nn.Conv2d(256, self.n_anchor_box, 1, 1, 0)

    def forward(self, x):
        reg = self.reg_conv1(x)
        reg = self.reg_conv2(reg)
        reg = reg.permute(0, 2, 3, 1)

        cls = self.cls_conv1(x)
        cls = self.cls_conv2(cls)
        cls = cls.permute(0, 2, 3, 1)

        return [reg, cls]
