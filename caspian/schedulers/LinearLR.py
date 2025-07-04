from . import Scheduler
from ..utilities import check_types

class LinearLR(Scheduler):
    """
    A linear changing learning rate scheduler. Takes a start and end position, and the number
    of expected epochs that it will run for. If it exceeds the expected epoch count, it will
    use the end rate for any epoch past the expected.

    Parameters
    ----------
    start_rate : float
        The initial learning rate that the scheduler will use at epoch 0.
    end_rate : float 
        The ending learning rate that the scheduler will use at the final epoch
        and beyond.
    max_iters : int
        The number of expected epoch for the scheduler to continuously modify the 
        learning rate.
    epoch : int
        The current step of the scheduler.
    """
    @check_types()
    def __init__(self, start_rate: float, end_rate: float, iters: int = 5) -> None:
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.max_iters = iters
        self.epoch = 0

    def __call__(self, *_) -> float:
        return self.start_rate - self.epoch * (self.start_rate - self.end_rate) / self.max_iters
    
    def __repr__(self) -> str:
        return f"LinearLR:{self.start_rate}:{self.end_rate}:{self.max_iters}"

    def step(self) -> None:
        self.epoch += 1 if self.epoch < self.max_iters else 0

    def reset(self) -> None:
        self.epoch = 0
    
    def deepcopy(self) -> 'LinearLR':
        return LinearLR(self.start_rate, self.end_rate, self.max_iters)