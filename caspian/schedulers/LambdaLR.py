from . import Scheduler

class LambdaLR(Scheduler):
    """
    A customizable function scheduler. Takes a function with the scheduler's step as the
    main input and multiplies it by the initial learning rate.


    Notes
    -----
    When saving and loading a layer or optimizer with this scheduler, you will need to 
    set the `self.funct` value of this scheduler back into the function you wish for it to
    use, otherwise it will not function properly.


    Parameters
    ----------
    funct : Callable
        A function which takes in the current epoch/step as the input and gives a float
        value back.
    epoch : int
        The epoch or step that the scheduler is currently on.
    """
    def __init__(self, funct: callable) -> None:
        self.funct = funct
        self.epoch = 0

    def __call__(self, init_rate: float) -> float:
        return init_rate * self.funct(self.epoch)
    
    def __repr__(self) -> str:
        return "LambdaLR:None"

    def step(self) -> None:
        self.epoch += 1

    def reset(self) -> None:
        self.epoch = 0

    def deepcopy(self) -> 'LambdaLR':
        return LambdaLR(self.funct)