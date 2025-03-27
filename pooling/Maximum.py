from cudalib import np
from . import PoolFunc

class Maximum(PoolFunc):
    def forward(self, partition: np.ndarray) -> np.ndarray:
        return np.max(partition, axis=self.axis)

    def backward(self, partition: np.ndarray) -> np.ndarray:
        #mask = np.zeros(partition.shape, dtype=partition.dtype)
        #mask[partition.argmax()] = 1
        mask = np.max(partition, axis=self.axis, keepdims=True) == partition
        return mask