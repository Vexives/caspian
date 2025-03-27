from . import Scheduler

class ConstantLR(Scheduler):
    """
    A constant multiplier learning rate scheduler. For every epoch up until the last expected epoch,
    the scheduler multiplies the initial learning rate by the given constant. Afterwards, it will
    just return the initial rate.

    Parameters
    ----------
    steps : int
        The number of epochs before each multiplicative step is taken.
    const : float
        The gamma value that the learning rate is multiplied by every step.
    epoch : int
        The epoch or step that the scheduler is currently on.
    """
    def __init__(self, steps: int, const: float = 0.1) -> None:
        self.steps = steps
        self.const = const
        self.epoch = 0

    def __call__(self, init_rate: float) -> float:
        return init_rate * self.const if self.epoch < self.steps else init_rate
    
    def __repr__(self) -> str:
        return f"ConstantLR:{self.steps}:{self.const}"

    def step(self) -> None:
        self.epoch += 1

    def reset(self) -> None:
        self.epoch = 0

    def deepcopy(self) -> 'ConstantLR':
        return ConstantLR(self.steps, self.const)