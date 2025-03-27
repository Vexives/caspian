from cudalib import np
from . import Optimizer
from schedulers import Scheduler, SchedulerLR

class StandardGD(Optimizer):
    """
    A standard gradient descent optimizer. The most basic of completed optimizers, multiplies
    the iteration's learn rate by the given gradient and returns.

    Parameters
    ----------
    learn_rate : float
        The initial learn rate that is given to this layer's scheduler.
    scheduler : Scheduler
        The learn rate scheduler that this optimizer wraps.
    """
    def __init__(self, learn_rate: float = 0.01, sched: Scheduler = SchedulerLR()) -> None:
        super().__init__(learn_rate, sched)
    
    def __repr__(self) -> str:
        return f"StandardGD/{self.learn_rate}/" + repr(self.scheduler)

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        learn_rate = self.scheduler(self.learn_rate)
        return -learn_rate * grad
        
    def step(self) -> None:
        self.scheduler.step()

    def reset_grad(self) -> None:
        self.scheduler.reset()

    def deepcopy(self) -> 'StandardGD':
        return StandardGD(self.learn_rate, self.scheduler.deepcopy())