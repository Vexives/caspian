class Scheduler():
    '''
    A basic learning rate scheduler container class which all Caspian schedulers inherit from.
    Any custom schedulers should inherit from this container class.

    Performs no operations and takes no arguments.
    '''
    def __init__(self) -> None:
        pass

    def __call__(self) -> float:
        pass

    def __repr__(self) -> str:
        pass

    def step(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def deepcopy(self) -> 'Scheduler':
        pass