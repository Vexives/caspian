from caspian.cudalib import np
from . import Activation

class Linear(Activation):
    """
    A linear activation function, applies no filter to the data before returning, and
    the gradient returned on the backwards pass is always constant.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "Linear"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return data
    
    def backward(self, _: np.ndarray) -> np.ndarray:
        return 1