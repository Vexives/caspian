from caspian.cudalib import np

class Loss():
    '''
    A basic loss container class which all Caspian loss inherit from.
    Any custom loss functions should inherit from this container class.
    
    Performs no operations and takes no arguments.
    '''
    def __call__(self, actual: np.ndarray, 
                 prediction: np.ndarray, backward: bool = False) -> np.ndarray:
        return self.backward(actual, prediction) if backward else self.forward(actual, prediction)
    
    @staticmethod
    def forward(actual: np.ndarray, prediction: np.ndarray) -> float:
        pass

    @staticmethod
    def backward(actual: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        pass