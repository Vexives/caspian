from ..cudalib import np
from . import Activation

class HardTanh(Activation):
    """
    A hard hyperbolic tangent activation function, applies `max(min(data, upper), lower)` to the input data.
    
    Backwards pass returns 1 if the data is between `upper` and `lower`, and 0 otherwise.

    Attributes
    ----------
    lower : float
        The lower bounding limit of this instance, any values in the array that are below it
        will be assigned this value.
    upper : float
        The upper bounding limit of this instance, any values in the array that are above it
        will be assigned this value.
    """
    def __init__(self, lower: float = -1.0, upper: float = 1.0):
        self.min = lower
        self.max = upper

    def __repr__(self) -> str:
        return f"HardTanh/{self.min}/{self.max}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.maximum(np.minimum(data, self.max), self.min)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return ((data < self.max) & (data > self.min)) * 1