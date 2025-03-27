from cudalib import np
from . import Loss

class MeanSquareErr(Loss):
    """
    A static class which gives both the forward and backward passes for the Mean Square Error (L2)
    loss function.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        return 0.5 * np.mean(np.square(actual - prediction))

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return (prediction - actual) / actual.size