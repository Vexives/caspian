from caspian.cudalib import np
from . import Loss

class MeanAvgErr(Loss):
    """
    A static class which gives both the forward and backward passes for the Mean Average Error (L1)
    loss function.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        return np.mean(np.absolute(actual - prediction))

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return np.sign(prediction - actual) / actual.size