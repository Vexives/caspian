from ..cudalib import np
from . import Loss

class Direct(Loss):
    """
    A static class which gives both the forward and backward passes for a direct loss function.
    A basic loss function, returns the prediction subtracted from the actual label.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        return np.sum(actual - prediction)

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return prediction - actual