import numpy as np
import torch
import torch.nn as nn

from darknet19 import DarkNet19
from rpn import RPN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class YOLOv2(nn.Module):
    def __init__(self):
        super(YOLOv2, self).__init__()
        self.darknet = DarkNet19()
        self.rpn = RPN()

    def forward(self, x):
        x = self.darknet(x)



def target_generator(bbox, class_, n_bbox, n_class, in_size, out_size):
    in_h, in_w = in_size[0], in_size[1]
    out_h, out_w = out_size[0], out_size[1]
    bbox_h, bbox_w = bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    bbox_y, bbox_x = bbox[:, 0] + .5 * bbox_h, bbox[:, 1] + .5 * bbox_w

    objs = torch.zeros(out_size, dtype=torch.long).to(device)
    ratio = out_h / in_h
    bbox_y1_warp, bbox_x1_warp = torch.floor(bbox[:, 0] * ratio).int(), torch.floor(bbox[:, 1] * ratio).int()
    bbox_y2_warp, bbox_x2_warp = torch.ceil(bbox[:, 2] * ratio).int(), torch.ceil(bbox[:, 3] * ratio).int()
    for i in range(n_bbox):
        objs[bbox_y1_warp[i]:bbox_y2_warp[i] + 1, bbox_x1_warp[i]:bbox_x2_warp[i] + 1] = 1

    target = torch.zeros((out_h, out_w, 5 * n_bbox + n_class)).to(device)
    for i in range(n_bbox):
        target[:, :, 5 * i:5 * i + 4] = torch.Tensor([bbox_y[i], bbox_x[i], bbox_h[i], bbox_w[i]])
        target[:, :, 5 * i + 4] = objs
        for j in range(4):
            target[:, :, 5 * i + j] *= objs
    for y_ in range(out_h):
        for x_ in range(out_w):
            for c in class_:
                target[y_, x_, 5 * n_bbox + c * objs[y_, x_]] = 1

    return target


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLOv1(20, 1).to(device)
    model.pretrain = False
    summary(model, (3, 224, 224))
