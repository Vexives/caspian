from caspian.cudalib import np
from . import Loss

class Huber(Loss):
    """
    A non-static class that performs the forward and backwards passes of the Huber loss function.
    Requires initialization.

    Parameters
    ----------
    delta : float
        The delta value of the equation to which the array's results will be compared to.
    """
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, actual: np.ndarray, prediction: np.ndarray) -> float:
        a_val = np.abs(actual - prediction)
        first_cond = 0.5 * np.square(actual - prediction)
        sec_cond = self.delta * (a_val - (0.5 * self.delta))
        return np.mean(np.where(a_val <= self.delta, first_cond, sec_cond))

    def backward(self, actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        a_val = np.abs(actual - prediction)
        first_cond = -(prediction - actual)
        sec_cond = -self.delta * np.sign(prediction - actual)
        return np.mean(np.where(a_val <= self.delta, first_cond, sec_cond))
