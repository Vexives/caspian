from ..cudalib import np
from . import Softmax

class Softmin(Softmax):
    """
    A Softmin activation function, creates an even distribution based on the 
    negated values of the data.

    Backwards pass returns the matrix multiplied result of the Jacobian gradient.


    Notes
    -----
    Backward pass intended to be used with the `CrossEntropy` loss type and its derivative.


    Attributes
    ----------
    axis : int
        The axis at which the softmax function is performed.
    """
    def __init__(self, axis: int = -1):
        self.axis = axis

    def __repr__(self) -> str:
        return f"Softmin/{self.axis}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return super().forward(-data)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        return data