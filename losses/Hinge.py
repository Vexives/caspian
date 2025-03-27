from cudalib import np
from . import Loss

class Hinge(Loss):
    """
    A static class which gives both the forward and backward passes for the Hinge
    loss function.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        return max(0, 1.0 - (actual * prediction))

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        deriv = actual * prediction
        return np.where(deriv < 1, -actual, 0)
