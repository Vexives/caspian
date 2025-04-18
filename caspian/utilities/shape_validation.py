from ..cudalib import np

def validate_input(funct):
    def wrapper(self, *args):
        minsize = min(len(self.in_size), len(args[0].shape))
        assert self.in_size[-minsize:] == args[0].shape[-minsize:], \
        f"Input shape does not match layer input shape. {self.in_size} - {args[0].shape}"
        return funct(self, *args)
    return wrapper


def validate_grad(funct):
    def wrapper(self, *args):
        minsize = min(len(self.out_size), len(args[0].shape))
        assert self.out_size[-minsize:] == args[0].shape[-minsize:], \
        f"Gradient shape does not match layer output shape. {self.in_size} - {args[0].shape}"
        return funct(self, *args)
    return wrapper


def confirm_shape(in_size: tuple[int, ...], expected: tuple[int, ...], indices: int):
    return len(in_size) <= indices+1 and in_size[-indices:] == expected


def all_ints(element: tuple | int) -> bool:
    if element is None:
        return True
    if not isinstance(element, tuple):
        return isinstance(element, int)
    return all(map(lambda x: isinstance(x, int), element))


def all_positive(contents: tuple | int | float, include_zero: bool = False) -> bool:
    try:
        if isinstance(contents, tuple):
            return all(map(lambda x: x >= int(not include_zero), contents))
        return contents >= int(not include_zero)
    except TypeError:
        return False