from ..cudalib import np
from . import Loss

class CrossEntropy(Loss):
    """
    A static class which gives both the forward and backward passes for the Categorical
    Cross-Entropy loss function.

    Does not initialize, and does not keep any parameters.
    """
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        clip_pred = np.clip(prediction, 1e-10, 1 - 1e-10)
        batch_div = prediction.shape[0] if len(prediction.shape) > 1 else 1
        return -np.sum(actual * np.log(clip_pred)) / batch_div

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        batch_div = prediction.shape[0] if len(prediction.shape) > 1 else 1
        return (prediction - actual) / batch_div