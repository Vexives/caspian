from ..cudalib import np
from . import Activation, Sigmoid

class Swish(Activation):
    """
    A Swish activation function, applies `data * sigmoid(data)` to the input data.

    Backwards pass applies `sigmoid(data) * data * (1 - sigmoid(data))` to the gradient.

    Attributes
    ----------
    beta : float
        A given float value representing the beta value which will be applied to the
        data before being passed into the `Sigmoid` function.
    """
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.__sigmoid = Sigmoid()

    def __repr__(self) -> str:
        return f"Swish/{self.beta}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.__last_sig = self.__sigmoid(self.beta * data)
        self.__last_in = data
        return data * self.__last_sig

    def backward(self, *_) -> np.ndarray:
        return self.__last_sig + self.__last_in * self.__last_sig * (1 - self.__last_sig)