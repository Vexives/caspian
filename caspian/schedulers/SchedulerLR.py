from . import Scheduler

class SchedulerLR(Scheduler):
    """
    The most basic of learning rate schedulers, does not change the initial learning rate
    and is a placeholder for schedulers which modify the rate value.

    Takes and holds no parameters.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, init_rate: float) -> float:
        return init_rate
    
    def __repr__(self) -> str:
        return "SchedulerLR"

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def deepcopy(self) -> 'SchedulerLR':
        return SchedulerLR()