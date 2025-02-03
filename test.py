import torch
print(torch.cuda.is_available())  # 应输出True
print(torch.version.cuda)         # 查看PyTorch依赖的CUDA版本