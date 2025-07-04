from ..cudalib import np
from ..schedulers import Scheduler, SchedulerLR
from ..utilities import check_types

class Optimizer():
    '''
    A basic optimizer container class which all Caspian optimizers inherit from.
    Any custom optimizers should inherit from this container class.

    Performs no operations and takes no arguments.
    '''
    @check_types()
    def __init__(self, learn_rate: float = 0.01, sched: Scheduler = SchedulerLR()) -> None:
        self.learn_rate = learn_rate
        self.scheduler = sched
        
    def __call__(self, grad: np.ndarray) -> np.ndarray:
        return self.process_grad(grad)

    def __repr__(self) -> str:
        pass

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        pass

    def step(self) -> None:
        pass

    def reset_grad(self) -> None:
        pass

    def deepcopy(self) -> 'Optimizer':
        pass