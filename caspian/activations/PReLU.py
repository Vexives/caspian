from ..cudalib import np
from . import Activation
from ..optimizers import Optimizer, StandardGD

class PReLU(Activation):
    """
    A Parameterized ReLU activation function, applies `max(0, data) + alpha * min(0, data)` to the input data.
    Contains weighted values that are applied on the axis of choice. Represents the value applied to all 
    negative values.
    
    Backwards pass returns 1 if the data is greater than 0, and `alpha` otherwise.
    Call the function with the last output and gradient in order for the weights to update.


    Notes
    -----
    This is a parameterized activation function, meaning any backward calls with gradients will update
    their values. The number of weights given should either be of size 1 or of the number of channels
    expected from any input arrays (at a chosen axis).


    Attributes
    ----------
    weights : ndarray
        An array of weights that represent the constants applied to negative values on the chosen axis.
    axis : int
        The axis at which the weights will be applied to.
    opt : Optimizer
        The chosen optimizer for this activation's weights, used when updating values.
    """
    def __init__(self, channels: int = 1, axis: int = 1, 
                 init: float = 0.25, optimizer: Optimizer = StandardGD()):
        self.weights = np.ones((channels,)) * init
        self.__init = init
        self.axis = axis
        self.opt = optimizer

    def __call__(self, data: np.ndarray, err: np.ndarray = None):
        if err is not None:
            new_grad = err * self.backward(data) 
            self.weights += self.opt(new_grad).swapaxes(self.axis, -1) \
                            .reshape(-1, self.weights.shape[0]).sum()
            return new_grad
        return self.forward(data)

    def __repr__(self) -> str:
        return f"PReLU/{self.weights.shape[0]}/{self.axis}/{self.__init}\u2007{repr(self.opt)}\u2007" + \
                " ".join(list(map(str, self.weights.flatten().tolist())))
    
    def step(self) -> None:
        self.opt.step()

    def reset_grad(self) -> None:
        self.opt.reset_grad()

    def forward(self, data: np.ndarray) -> np.ndarray:
        reshaped_data = data.swapaxes(self.axis, -1)
        return (np.maximum(0, reshaped_data) + (self.weights * np.minimum(0, reshaped_data))).swapaxes(self.axis, -1)

    def backward(self, data: np.ndarray) -> np.ndarray:
        reshaped_data = data.swapaxes(self.axis, -1)
        return np.minimum(1, self.weights + (reshaped_data > 0) * 1).swapaxes(self.axis, -1)