import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DarkNet19(nn.Module):
    def __init__(self, cfgs=None):
        super(DarkNet19, self).__init__()
        self.cfgs = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256, 'M', 512, 256, 512, 256, 512, 'M', 1024, 512, 1024, 512, 1024] \
                    if cfgs is None else cfgs
        self.conv_blocks1, self.conv_blocks2 = DarkNet19.make_conv_blocks(self.cfgs)
        # self.conv1 = nn.Conv2d(1024, 1000, 1, 1, 0)
        # self.gap = DarkNet19.global_average_pooling
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        fine_grained = self.conv_blocks1(x)
        x = self.conv_blocks2(fine_grained)

        fine_grained = fine_grained.view((-1, 2048, 13, 13))
        x = torch.cat([x, fine_grained], dim=1)

        return x

    @staticmethod
    def make_conv_blocks(cfgs):
        layers1, layers2 = [], []
        cfg_prev = 3
        fine_grained = False
        layers = layers1
        dummy = torch.Tensor(2, 3, 416, 416).to(device)
        for cfg in cfgs:
            if cfg == 'M':
                layers.append(nn.BatchNorm2d(cfg_prev))
                layers.append(nn.LeakyReLU(inplace=True))
                layers.append(nn.MaxPool2d(2, 2))
            else:
                if cfg_prev < cfg:
                    layers.append(nn.Conv2d(cfg_prev, cfg, 3, 1, 1).to(device))
                else:
                    layers.append(nn.Conv2d(cfg_prev, cfg, 1, 1, 0).to(device))
                cfg_prev = cfg
            dummy = layers[-1](dummy)
            if not fine_grained and dummy.shape == (2, 512, 26, 26):
                fine_grained = True
                layers = layers2

        return nn.Sequential(*layers1), nn.Sequential(*layers2)



    @staticmethod
    def global_average_pooling(x):
        return x.mean(dim=(2, 3))


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DarkNet19().to(device)
    summary(model, (3, 416, 416))
    dummy = torch.Tensor(2, 3, 416, 416).to(device)
    output = model(dummy)
    print(output.shape)