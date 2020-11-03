import torch
import torch.nn as nn


class DarkNet19(nn.Module):
    def __init__(self, cfgs):
        super(DarkNet19, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(128, 64, 1, 0, 0)
        self.conv3_3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(256, 128, 1, 0, 0)
        self.conv4_3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 256, 1, 0, 0)
        self.conv5_3 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv5_4 = nn.Conv2d(512, 256, 1, 0, 0)
        self.conv5_5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6_1 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv6_2 = nn.Conv2d(1014, 512, 1, 0, 0)
        self.conv6_3 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv6_4 = nn.Conv2d(1024, 512, 1, 0, 0)
        self.conv6_5 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.conv7 = nn.Conv2d(1024, 1000, 1, 0, 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.gap = DarkNet19.global_average_pooling
        self.leakyrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.covn3_3(x)
        x = self.bn3(x)
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.bn4(x)
        x = self.maxpool(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.bn5(x)
        x = self.maxpool(x)

        x = self.


    @staticmethod
    def make_conv_blocks(cfgs):
        layers = []
        cfg_prev = 3
        for cfg in cfgs:
            if cfg == 'M':
                layers.append(nn.BatchNorm2d(cfg_prev))
                layers.append(nn.MaxPool2d(2, 2))
            else:
                if cfg_prev < cfg:
                    layers.append(nn.Conv2d(cfg_prev, cfg, 3, 1, 1))
                else:
                    layers.append(nn.Conv2d(cfg_prev, cfg, 1, 0, 0))
            cfg_prev = cfg



    @staticmethod
    def global_average_pooling(x):
        return x.mean(dim=(2, 3))
