import torch

n, c, h, w = 10000, 3, 256, 256
data_file = str(n) + "_random_chw_tensors.npz"

t = torch.rand(n, c, h, w)

torch.save(t, data_file)