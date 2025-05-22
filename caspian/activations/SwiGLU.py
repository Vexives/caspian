from ..cudalib import np
from . import Activation, Swish
from ..utilities import InvalidDataException

class SwiGLU(Activation):
    """
    A Swish-GLU activation function, splits the data into two parts, adding a `Swish` function
    to the second half. Performed across a given axis.
    
    Backwards pass returns a concatenated gradient of the initial split input.

    
    Notes
    -----
    This activation function is unique in that it does not return the same number of elements
    in its output as it does the input. The axis that is chosen is split in half, and MUST be 
    an even number before passing in.


    Attributes
    ----------
    axis : int
        A given integer value representing the axis that the data will be split by.
    beta : float
        A given float value representing the beta value which will be applied to the
        data during the `Swish` pass.
    """
    def __init__(self, axis: int = -1, beta: float = 1.0):
        self.beta = beta
        self.axis = axis
        self.__swish = Swish(self.beta)
    
    def __repr__(self) -> str:
        return f"SwiGLU/{self.axis}/{self.beta}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.__first, self.__second = np.split(data, 2, self.axis)
        if self.__first.shape != self.__second.shape:
            raise InvalidDataException("Shape for selected axis of data must be even.")
        return self.__first * self.__swish(self.__second)

    def backward(self, *_) -> np.ndarray:
        back_swish = self.__swish(self.__second)
        return np.concatenate((back_swish, self.__first * self.__swish(back_swish, True)), self.axis)