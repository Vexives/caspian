from ..cudalib import np
from . import Activation
from ..utilities import InvalidDataException, check_types

class RReLU(Activation):
    """
    A Randomized ReLU activation function, applies `max(data * U, data)` to the input data,
    where `U` is a randomized array sampled from the uniform distribution.
    
    Backwards pass returns 1 if the data is greater than 0, and alpha otherwise.

    Attributes
    ----------
    lower : float
        A given float value representing the lower bound of values to be sampled from randomly.
    upper : float
        A given float value representing the upper bound of values to be sampled from randomly.
    """
    @check_types()
    def __init__(self, lower: float = 0.125, upper: float = 1.0/3.0):
        if lower > upper:
            raise InvalidDataException("Argument \"lower\" must be greater than argument \"upper\".")
        self.lower = lower
        self.upper = upper

    def __repr__(self) -> str:
        return f"RReLU/{self.lower}/{self.upper}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.__rand_mask = np.random.uniform(self.lower, self.upper, data.shape)
        return np.where(data >= 0, data, data * self.__rand_mask)
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        return np.where(data > 0, 1, self.__rand_mask)