import torch
import numpy as np


print("PyTorch loaded from:", torch.__file__)
print("NumPy loaded from:", np.__file__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Torch version: ", torch.__version__)
