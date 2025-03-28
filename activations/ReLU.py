from caspian.cudalib import np
from . import Activation

class ReLU(Activation):
    """
    A ReLU activation function, applies `max(0, data)` to the input data.
    
    Backwards pass returns 1 if the data is greater than 0, and 0 otherwise.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "ReLU"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(0, data)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return (data >= 0) * 1