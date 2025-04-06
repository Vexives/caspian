from ..cudalib import np
from . import PoolFunc

class Minimum(PoolFunc):
    def forward(self, partition: np.ndarray) -> np.ndarray:
        return np.min(partition, axis=self.axis)
    
    def backward(self, partition: np.ndarray) -> np.ndarray:
        #mask = np.zeros(partition.shape, dtype=partition.dtype)
        #mask[partition.argmin()] = 1
        mask = np.min(partition, axis=self.axis, keepdims=True) == partition
        return mask