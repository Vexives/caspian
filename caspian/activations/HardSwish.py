from ..cudalib import np
from . import Activation, ReLUX

class HardSwish(Activation):
    """
    A hard swish activation function, applies `data * RELU6(data)/6` to the input data.
    
    Backwards pass returns `2 * data / 6` if between -3.0 & 3.0, 0 if below -3.0, and 1 if above 3.0.

    Takes and holds no parameters.
    """
    def __repr__(self) -> str:
        return "HardSwish"

    def forward(self, data: np.ndarray) -> np.ndarray:
        return data * np.minimum(np.maximum(0, data+3), 6)/6

    def backward(self, data: np.ndarray) -> np.ndarray:
        new_data = (data > 0) * 1
        return np.where((data >= -3) & (data <= 3), (2*data + 3)/6, new_data)