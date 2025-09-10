from ..optimizers import parse_opt_info
from ..cudalib import np

from .Activation import Activation
from .ReLU import ReLU
from .Sigmoid import Sigmoid
from .Identity import Identity
from .Tanh import Tanh
from .Softplus import Softplus
from .LReLU import LReLU
from .ELU import ELU
from .RReLU import RReLU
from .Swish import Swish
from .Softsign import Softsign
from .GLU import GLU
from .SwiGLU import SwiGLU
from .Softmax import Softmax
from .Softmin import Softmin
from .ReLUX import ReLUX
from .HardShrink import HardShrink
from .HardTanh import HardTanh
from .HardSwish import HardSwish
from .HardSigmoid import HardSigmoid
from .PReLU import PReLU

act_funct_dict: dict[str, Activation] = {"ReLU":ReLU, 
                                         "Sigmoid":Sigmoid, 
                                         "Tanh":Tanh, 
                                         "Softmax":Softmax,
                                         "LReLU":LReLU, 
                                         "Softplus":Softplus, 
                                         "Softmin":Softmin,
                                         "Softsign":Softsign,
                                         "Swish":Swish,
                                         "ELU":ELU,
                                         "RReLU":RReLU,
                                         "GLU":GLU,
                                         "SwiGLU":SwiGLU,
                                         "ReLUX":ReLUX,
                                         "Hardshrink":HardShrink,
                                         "HardTanh":HardTanh,
                                         "HardSwish":HardSwish,
                                         "HardSigmoid":HardSigmoid,
                                         "PReLU":PReLU,
                                         "Identity":Identity}

param_funct_dict: dict[str, Activation] = {"PReLU":PReLU}


def parse_act_info(input: str) -> Activation:
    all_params = input.strip().split("/")
    if all_params[0] in param_funct_dict:   # Parameterized activations
        return __parse_param_act_info(input)
    if all_params[0] not in act_funct_dict:
        return Activation()
    
    format_params = list(map(__map_to_numeric, all_params[1:]))
    return act_funct_dict[all_params[0]](*format_params)

def __parse_param_act_info(input: str) -> Activation:
    all_lines = input.strip().split("\u2007")   # Param-act separator
    all_params = all_lines[0].split("/")

    format_params = list(map(__map_to_numeric, all_params[1:]))
    new_opt = parse_opt_info(all_lines[1])
    weights = np.array(list(map(float, all_lines[-1].split())))

    new_act = param_funct_dict[all_params[0]](*format_params)
    new_act.weights = weights
    new_act.opt = new_opt
    return new_act

def __map_to_numeric(input: str) -> tuple | int:
    try:
        return int(input)
    except:
        return float(input)