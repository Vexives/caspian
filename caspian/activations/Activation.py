from ..cudalib import np

class Activation():
    '''
    A basic activation container class which all Caspian activations inherit from.
    Any custom activation functions should inherit from this container class.
    
    Performs no operations and takes no arguments.
    '''
    def __call__(self, data: np.ndarray, err: np.ndarray = None) -> np.ndarray:
        return err * self.backward(data) if err is not None else self.forward(data)
    
    def __repr__(self) -> str:
        return "Custom"

    def forward(self, data: np.ndarray) -> np.ndarray:
        pass

    def backward(self, data: np.ndarray) -> np.ndarray:
        pass