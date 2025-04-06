from ..cudalib import np
from . import Optimizer
from ..schedulers import Scheduler, SchedulerLR

class Momentum(Optimizer):
    """
    A momentum-based optimizer which includes the momentum and velocity of a moving gradient.

    Parameters
    ----------
    momentum : float
        The rate of momentum that is kept between gradient passes. Normally between 0.0 and 1.0.
    learn_rate : float
        The initial learn rate that is given to this layer's scheduler.
    scheduler : Scheduler
        The learn rate scheduler that this optimizer wraps.
    """
    def __init__(self, momentum: float = 0.9, learn_rate: float = 0.01,
                 sched: Scheduler = SchedulerLR()) -> None:
        super().__init__(learn_rate, sched)
        self.momentum = momentum
        self.__previous = 0.0
    
    def __repr__(self) -> str:
        return f"Momentum/{self.momentum}/{self.learn_rate}/" + repr(self.scheduler)

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        learn_rate = self.scheduler(self.learn_rate)
        velocity_grad = self.momentum * self.__previous - learn_rate * grad
        self.__previous = velocity_grad
        return velocity_grad
    
    def step(self) -> None:
        self.scheduler.step()
    
    def reset_grad(self) -> None:
        self.__previous = 0.0
        self.scheduler.reset()

    def deepcopy(self) -> 'Momentum':
        return Momentum(self.momentum, self.learn_rate, self.scheduler.deepcopy())