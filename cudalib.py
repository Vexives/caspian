import os
cuda_support = os.getenv("CSPN_CUDA")

if cuda_support == "cuda":
    try:
        import cupy as np
    except Exception as err:
        print(f"Caspian -> CUDA Import Error:\n\"{err}\"\n" +
               "Defaulting to CPU compute (NumPy).")
        import numpy as np
else:
    import numpy as np