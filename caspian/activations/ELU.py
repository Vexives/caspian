from ..cudalib import np
from . import Activation
from ..utilities import check_types

class ELU(Activation):
    """
    An ELU activation function, applies `where(data > 0, data, alpha * (e^data - 1))` 
    to the input data.
    
    Backwards pass returns 1 if the data is greater than 0, and `data + alpha` otherwise.

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
        return f"ELU/{self.alpha}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.where(data > 0, data, self.alpha * (np.exp(data) - 1))

    def backward(self, data: np.ndarray) -> np.ndarray:
        return np.where(data > 0, 1, data + self.alpha)