from ..cudalib import np
from . import Activation
from ..utilities import check_types

class LReLU(Activation):
    """
    A Leaky ReLU activation function, applies `max(data * alpha, data)` to the input data.
    
    Backwards pass returns 1 if the data is greater than 0, and alpha otherwise.

    Attributes
    ----------
    alpha : float
        A given float value representing the alpha value which any negative values
        will be multiplied by.
    """
    @check_types()
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def __repr__(self) -> str:
        return f"LReLU/{self.alpha}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.where(data >= 0, data, data * self.alpha)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        return np.where(data > 0, 1, self.alpha)