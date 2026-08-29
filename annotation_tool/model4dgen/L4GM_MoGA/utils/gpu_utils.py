import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")