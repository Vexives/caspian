from ..cudalib import np
from . import Activation

class HardSigmoid(Activation):
    """
    A hard sigmoid activation function, applies `x/6 + 0.5` to the input data if between -3.0 & 3.0,
    0 if below -3.0, and 1 if above 3.0.
    
    Backwards pass returns 1/6 if between values of -3.0 & 3.0, 0 otherwise.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "HardSigmoid"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(np.minimum(0.5 + data/6, 1), 0)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return ((data <= 3) & (data >= -3)) * 1.0/6