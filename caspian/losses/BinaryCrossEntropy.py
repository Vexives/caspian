from ..cudalib import np
from . import Loss

class BinaryCrossEntropy(Loss):
    """
    A static class which gives both the forward and backward passes for the Negative Log Likelihood
    loss function.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        return -np.mean((actual * np.log(prediction)) + ((1 - actual) * np.log(1 - prediction)))

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return prediction - actual