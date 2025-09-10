from ..cudalib import np
from . import Activation

class Softsign(Activation):
    """
    A soft-sign activation function, applies `data / (1 + abs(data))` to the input data.

    Backwards pass applies `1 / (1 + abs(data))^2` to the gradient.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "Softsign"

    def forward(self, data: np.ndarray) -> np.ndarray:
        self.__last_abs = (1 + np.abs(data))
        return (data / self.__last_abs)

    def backward(self, *_) -> np.ndarray:
        return 1 / np.square(self.__last_abs)