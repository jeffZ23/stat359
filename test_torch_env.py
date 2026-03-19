import torch
import scipy
import numpy
import gensim

print("PyTorch version:", torch.__version__)
print("scipy version:", scipy.__version__)
print("numpy version:", numpy.__version__)
print("gensim version:", gensim.__version__)

if torch.cuda.is_available():
    print("PyTorch is using GPU:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("PyTorch is using Apple Silicon MPS (Metal Performance Shaders)")
    device = torch.device("mps")
else:
    print("PyTorch is using CPU")
    device = torch.device("cpu")

# Simple torch array operation
a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)
result = a + b
print("a:", a)
print("b:", b)
print("a + b:", result)
