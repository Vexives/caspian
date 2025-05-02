from ..cudalib import np
from . import InvalidDataException
import inspect
from types import UnionType
from typing import get_origin, get_args

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
    except:
        return False
    

def check_types(rules: list[tuple[str, callable, str]] = None) -> callable:
    rules = rules or []
    def parse_union(uniontype) -> UnionType:
        # Set origin to every subclass
        new_args = tuple(get_origin(arg) or arg 
                         for arg in get_args(uniontype))
        return new_args

    def decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs) -> object:
            # Get signature for every argument
            sign = inspect.signature(func)
            ls = sign.bind(*args, **kwargs)
            ls.apply_defaults()
            arg_dict = dict(ls.arguments.items())
                
            # Type-checking and validation
            for k, v in arg_dict.items():
                arg_type = func.__annotations__.get(k, object)
                if get_origin(arg_type) is UnionType:
                    arg_type = parse_union(arg_type)
                else:
                    arg_type = get_origin(arg_type) or arg_type

                if not isinstance(v, arg_type):
                    raise InvalidDataException(f"Argument \"{k}\" expected type " + \
                                               f"{arg_type}, received {type(v)}")
                
            # Custom error handling with unique error messages
            for v in rules:
                temp_val = arg_dict[v[0]]
                if not v[1](temp_val):
                    raise InvalidDataException(v[2])

            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator