from ..cudalib import np
from ..optimizers import Optimizer
from ..utilities import all_ints, check_types, InvalidDataException

class Layer():
    '''
    A basic layer container class which all Caspian network layers inherit from.
    Any custom layers should inherit from this container class.
    
    Performs no operations and takes no arguments.
    '''
    @check_types(("in_size", all_ints, "Incorrect input shape type - Must be all integers."),
                 ("out_size", all_ints, "Incorrect output shape type - Must be all integers."))
    def __init__(self, in_size: tuple[int, ...] | None, out_size: tuple[int, ...] | None):
        self.in_size = in_size
        self.out_size = out_size

    def __call__(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        """Calls the class forward function and provides the given parameters."""
        return self.forward(data, training)
    
    def forward(self, data: np.ndarray, training: bool = False) -> np.ndarray:
        pass

    def backward(self, cost_err: np.ndarray) -> np.ndarray:
        pass

    def step(self) -> None:
        pass

    def clear_grad(self) -> None:
        pass

    def set_optimizer(self, opt: Optimizer = Optimizer()) -> None:
        pass

    def deepcopy(self) -> 'Layer':
        pass

    def save_to_file(self, filename: str = None) -> None | str:
        pass

    def from_save(context: str, file_load: bool = False) -> 'Layer':
        pass
