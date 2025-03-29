from caspian.cudalib import np
from . import Optimizer
from caspian.schedulers import Scheduler, SchedulerLR

class ADAM(Optimizer):
    """
    The ADAM optimizer, combining learning strategies from `RMSProp` and `ADAGrad` optimizers for
    a highly efficient learning strategy.

    
    Parameters
    ----------
    d_one : float
        The first decay rate for moment estimates and bias corrected moment estimates. Normally
        close to 1.0.
    d_two : float
        The second decay rate for moment estimates and bias corrected moment estimates. Normally
        close to 1.0 and larger than `decay_one`.
    eps : float
        A very small float value that is added to the concentrated gradient before the 
        square root is taken.
    learn_rate : float
        The initial learn rate that is given to this layer's scheduler.
    scheduler : Scheduler
        The learn rate scheduler that this optimizer wraps.
    m : float | ndarray
        The first (biased) moment estimate for a gradient pass. First initialized to 0.0, but
        becomes a decaying concentrated estimate (`d_one`) as more gradients are processed.
    v : float | ndarray
        The second (biased) moment estimate for a gradient pass. First initialized to 0.0, but
        becomes a decaying concentrated estimate (`d_two`) as more gradients are processed.
    iter : int
        The epoch or iteration of the optimizer.
    """
    def __init__(self, decay_one: float = 0.9, decay_two: float = 0.999, eps: float = 1e-8,
                 learn_rate: float = 0.01, sched: Scheduler = SchedulerLR()) -> None:
        self.learn_rate = learn_rate
        self.scheduler = sched
        self.d_one = decay_one
        self.d_two = decay_two
        self.eps = eps
        self.m = 0.0
        self.v = 0.0
        self.iter = 1
    
    def __repr__(self) -> str:
        return f"ADAM/{self.d_one}/{self.d_two}/{self.eps}/{self.learn_rate}/" + repr(self.scheduler)

    def process_grad(self, grad: np.ndarray) -> np.ndarray:
        learn_rate = self.scheduler(self.learn_rate)
        
        #Moment Estimates
        self.m = self.d_one * self.m + (1 - self.d_one) * grad
        self.v = self.d_two * self.v + (1 - self.d_two) * grad**2

        #Bias-Corrected Moment Estimates
        m_hat = self.m / (1 - self.d_one**self.iter)
        v_hat = self.v / (1 - self.d_two**self.iter)

        #Final RMS Stage
        new_grad = (-learn_rate * m_hat) / (np.sqrt(v_hat) + self.eps)
        return new_grad
    
    def step(self) -> None:
        self.iter += 1
        self.scheduler.step()
    
    def reset_grad(self) -> None:
        self.m = 0.0
        self.v = 0.0
        self.iter = 1
        self.scheduler.reset()

    def deepcopy(self) -> 'ADAM':
        return ADAM(self.d_one, self.d_two, self.eps, self.learn_rate, self.scheduler.deepcopy())