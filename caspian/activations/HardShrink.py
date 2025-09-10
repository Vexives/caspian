from ..cudalib import np
from . import Activation
from ..utilities import check_types

class HardShrink(Activation):
    """
    A hard shrink activation function, applies `-delta < data < delta = 0` to the input data.
    
    Backwards pass returns 0 if the data is between `-delta` and `delta`, and 1 otherwise.

    Attributes
    ----------
    delta : float
        A float value representing the absolute range that will be shrunk down to zero from both
        the positive and negative values.
    """
    @check_types()
    def __init__(self, delta: float = 0.5):
        self.delta = delta

    def __repr__(self) -> str:
        return f"HardShrink/{self.delta}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return np.where((data > -self.delta) & (data < self.delta), 0, data)

    def backward(self, data: np.ndarray) -> np.ndarray:
        return (data != 0) * 1