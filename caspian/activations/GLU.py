from ..cudalib import np
from . import Activation, Sigmoid
from ..utilities import InvalidDataException, check_types

class GLU(Activation):
    """
    A GLU activation function, splits the data into two parts, adding a `Sigmoid` function
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
    """
    @check_types()
    def __init__(self, axis: int = -1):
        self.axis = axis
        self.__sigmoid = Sigmoid()
    
    def __repr__(self) -> str:
        return f"GLU/{self.axis}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.__first, self.__second = np.split(data, 2, self.axis)
        if self.__first.shape != self.__second.shape:
            raise InvalidDataException("Shape for selected axis of data must be even.")
        return self.__first * self.__sigmoid(self.__second)

    def backward(self, *_) -> np.ndarray:
        back_sig = self.__sigmoid(self.__second)
        return np.concatenate((back_sig, self.__first * self.__sigmoid(back_sig, True)), self.axis)