import torch


if __name__ == '__main__':
    a = torch.ones((3, 3))
    b = a.mean(dim=(0, 2))
    print(b)