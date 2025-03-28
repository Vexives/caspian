from caspian.cudalib import np

class PoolFunc():
    '''
    A basic pooling function container class which all Caspian pooling functions inherit from.
    Any custom functions should inherit from this container class.
    
    Performs no operations and takes no arguments.
    '''
    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, partition: np.ndarray, backward: bool = False) -> np.ndarray:
        return self.backward(partition) if backward else self.forward(partition)
    
    def __repr__(self):
        return f"{self.__class__.__name__}/{self.axis}"

    def forward(self, data: np.ndarray) -> np.ndarray:
        pass
    
    def backward(self, data: np.ndarray) -> np.ndarray:
        pass