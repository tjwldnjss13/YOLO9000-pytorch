import torch
import numpy as np

a = np.array([[1, 2, 3]])
b = np.array([[11, 22, 33]])

c = np.concatenate([a, b], axis=0)
print(c)
