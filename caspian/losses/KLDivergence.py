from ..cudalib import np
from . import Loss

class KLDivergence(Loss):
    """
    A static class which gives both the forward and backward passes for the Kullback-Leibler
    Divergence loss function.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        return np.sum(actual * np.log(actual / prediction))

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        return 1 + np.log(actual / prediction)