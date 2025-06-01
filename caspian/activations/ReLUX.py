from ..cudalib import np
from . import Activation

class ReLUX(Activation):
    """
    A ReLU-X activation function, applies `min(max(0, data), X)` to the input data.
    `X` is a given maximum value provided at initialization.
    
    Backwards pass returns 1 if the data is greater than 0 and below `X`, 0 otherwise.

    Attributes
    ----------
    x : float
        The value that determines the maximum value from this instance, default value is 6.0.
    """
    def __init__(self, x: float = 6.0):
        self.x = x

    def __repr__(self) -> str:
        return f"ReLUX/{self.x}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(0, data), self.x)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return ((data < self.x) & (data > 0)) * 1