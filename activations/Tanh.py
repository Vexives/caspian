from caspian.cudalib import np
from . import Activation

class Tanh(Activation):
    """
    A hyperbolic tangent activation function, applies `2 / (1 + exp(-2 * data))` to the input data.
    
    Backwards pass applies `1 - (grad)^2` to the gradient.

    Parameters
    ----------
    alpha : float
        A given float value representing the alpha value which any negative values
        will be multiplied by.
    """
    def __repr__(self) -> str:
        return "Tanh"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return (2 / (1 + np.exp(-2 * data))) - 1

    def backward(self, data: np.ndarray) -> np.ndarray:
        return 1 - np.square(data)