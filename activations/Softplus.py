from cudalib import np
from . import Activation

class Softplus(Activation):
    """
    A softplus activation function, applies `log(1 + exp(data))` to the input data.
    
    Backwards pass returns the `Sigmoid` forward pass of the gradient.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "Softplus"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(data))

    def backward(self, data: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-data)))