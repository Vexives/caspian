from ..cudalib import np
from . import Activation
from ..utilities import check_types

class Softmax(Activation):
    """
    A Softmax activation function, creates an even distribution based on the values of the data.

    Backwards pass returns the matrix multiplied result of the Jacobian gradient.


    Notes
    -----
    Backward pass intended to be used with the `CrossEntropy` loss type and its derivative.


    Attributes
    ----------
    axis : int
        The axis at which the softmax function is performed.
    """
    def __call__(self, data: np.ndarray, err: np.ndarray = None):
        return (err - (err * data).sum(axis=self.axis, keepdims=True)) * data \
               if err is not None else self.forward(data)

    @check_types()
    def __init__(self, axis: int = -1):
        self.axis = axis

    def __repr__(self) -> str:
        return f"Softmax/{self.axis}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        ex = np.exp(data - np.max(data, axis=self.axis, keepdims=True))
        self.__last_in = ex / ex.sum(axis=self.axis, keepdims=True)
        return self.__last_in
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        return data