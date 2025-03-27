import os
def enable_cuda():
    """Use this function before any other imports to enable CUDA/CuPy support for Caspian."""
    os.environ["CSPN_CUDA"] = "cuda"