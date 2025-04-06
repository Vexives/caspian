from ..cudalib import np
from . import PoolFunc

class Average(PoolFunc):
    def forward(self, partition: np.ndarray) -> np.ndarray:
        return np.average(partition, axis=self.axis)
    
    def backward(self, partition: np.ndarray) -> np.ndarray:
        return partition * (1.0 / partition.shape[self.axis])