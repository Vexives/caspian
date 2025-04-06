from ..cudalib import np
from . import Activation

class Softmin(Activation):
    """
    A Softmin activation function, creates an even distribution based on the 
    negated values of the data.

    Backwards pass returns the input gradient assuming that Cross Entropy loss is used.


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
        data = -data
        ex = np.exp(data - np.max(data, axis=self.axis, keepdims=True))
        return ex / ex.sum(axis=self.axis, keepdims=True)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        return data