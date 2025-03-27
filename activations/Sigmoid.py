from cudalib import np
from . import Activation

class Sigmoid(Activation):
    """
    A sigmoid activation function, applies `1 / 1 + exp(-data)` to the input data.

    Backwards pass applies `grad * (1 - grad)` to the gradient.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "Sigmoid"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-data)))

    def backward(self, data: np.ndarray) -> np.ndarray:
        return data * (1 - data)