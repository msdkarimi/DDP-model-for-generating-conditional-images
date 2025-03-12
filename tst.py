import torch
b = 4
i = 10
ts = torch.full((b,), i, dtype=torch.long)
print(ts.shape)
print(ts)