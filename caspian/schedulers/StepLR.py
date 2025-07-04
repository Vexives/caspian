from . import Scheduler
from ..utilities import check_types

class StepLR(Scheduler):
    """
    A step-wise learning rate scheduler. For every selected amount of epoch, this scheduler
    will create a multiplicative step in the learning rate, reducing or increasing it by the given
    gamma value.

    Parameters
    ----------
    steps : int
        The number of epochs before each multiplicative step is taken.
    gamma : float
        The gamma value that the learning rate is multiplied by every step.
    epoch : int
        The epoch or step that the scheduler is currently on.
    """
    @check_types()
    def __init__(self, steps: int, gamma: float = 0.1) -> None:
        self.steps = steps
        self.gamma = gamma
        self.epoch = 0

    def __call__(self, init_rate: float) -> float:
        return init_rate * (self.gamma ** (self.epoch // self.steps))
    
    def __repr__(self) -> str:
        return f"StepLR:{self.steps}:{self.gamma}"

    def step(self) -> None:
        self.epoch += 1

    def reset(self) -> None:
        self.epoch = 0

    def deepcopy(self) -> 'StepLR':
        return StepLR(self.steps, self.gamma)