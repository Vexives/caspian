from ..cudalib import np
from . import Optimizer
from ..schedulers import Scheduler, SchedulerLR

class RMSProp(Optimizer):
    """
    A Root Mean Squared Propagation optimizer. Applies a decay rate and a concentration of
    previous gradients to the current gradient before multiplying by the learn rate and dividing
    by the square root of the concentrated gradient sum.

    Parameters
    ----------
    decay : float
        The decay rate for the concentrated gradient and the inverse of the rate that the new
        gradient will be added to the concentrated. Should be above 0.0 and below 1.0.
    eps : float
        A very small float value that is added to the concentrated gradient before the 
        square root is taken.
    learn_rate : float
        The initial learn rate that is given to this layer's scheduler.
    scheduler : Scheduler
        The learn rate scheduler that this optimizer wraps.
    """
    def __init__(self, decay: float = 0.9, eps: float = 1e-8,
                 learn_rate: float = 0.01, 
                 sched: Scheduler = SchedulerLR()) -> None:
        super().__init__(learn_rate, sched)
        self.decay = decay
        self.eps = eps
        self.__conc_grad = 0.0
    
    def __repr__(self) -> str:
        return f"RMSProp/{self.decay}/{self.eps}/{self.learn_rate}/" + repr(self.scheduler)

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        learn_rate = self.scheduler(self.learn_rate)
        self.__conc_grad = (self.decay * self.__conc_grad) + (1 - self.decay) * grad**2
        new_grad = (-learn_rate * grad) / (np.sqrt(self.__conc_grad + self.eps))
        return new_grad
    
    def step(self) -> None:
        self.scheduler.step()
    
    def reset_grad(self) -> None:
        self.__conc_grad = 0.0
        self.scheduler.reset()

    def deepcopy(self) -> 'RMSProp':
        return RMSProp(self.decay, self.eps, self.learn_rate, self.scheduler.deepcopy())