from cudalib import np
from . import Optimizer
from schedulers import Scheduler, SchedulerLR

class ADAGrad(Optimizer):
    """
    An ADAGrad optimizer. Applies a concentration of previous squared gradients to the current 
    gradient before multiplying by the learn rate and dividing by the square root of the 
    concentrated gradient sum.

    Parameters
    ----------
    eps : float
        A very small float value that is added to the concentrated gradient before the 
        square root is taken.
    learn_rate : float
        The initial learn rate that is given to this layer's scheduler.
    scheduler : Scheduler
        The learn rate scheduler that this optimizer wraps.
    conc_grad : float | ndarray
        The decaying concentrated gradient of prior gradient inputs. Initialized to 0.0 on first
        iteration.
    """
    def __init__(self, eps: float = 1e-8, learn_rate: float = 0.01,
                 sched: Scheduler = SchedulerLR()) -> None:
        super().__init__(learn_rate, sched)
        self.conc_grad = 0.0
        self.eps = eps
    
    def __repr__(self) -> str:
        return f"ADAGrad/{self.eps}/{self.learn_rate}/" + repr(self.scheduler)

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        learn_rate = self.scheduler(self.learn_rate)
        self.conc_grad += grad**2
        new_grad = (-learn_rate * grad) / (np.sqrt(self.conc_grad + self.eps))
        return new_grad
    
    def step(self) -> None:
        self.scheduler.step()
    
    def reset_grad(self) -> None:
        self.conc_grad = 0.0
        self.scheduler.reset()

    def deepcopy(self) -> 'ADAGrad':
        return ADAGrad(self.eps, self.learn_rate, self.scheduler.deepcopy())