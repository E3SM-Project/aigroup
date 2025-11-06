"""
Some computations in pytorch
"""

import torch
import numpy as np

# Print torch info on module import
print(f"[Python] Module loaded - torch version: {torch.__version__}")
print(f"[Python] CUDA available: {torch.cuda.is_available()}")


def compute_tensor_operation(data, size):
    """
    Perform a PyTorch tensor operation
    """
    print(f"[Python] Computing tensor operation on {size} elements")

    tensor = torch.from_numpy(np.array(data, dtype=np.float32))

    print(f"[Python] Input tensor: {tensor}")

    result_value = torch.sum(tensor).item()

    print(f"[Python] Result: {result_value}")

    return result_value
