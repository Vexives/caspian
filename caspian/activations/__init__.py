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
                                         "Identity":Identity}

def parse_act_info(input: str) -> Activation:
    all_params = input.strip().split("/")
    if all_params[0] not in act_funct_dict:
        return Activation()
    
    for param in all_params[1:]:
        if param.find('.') != -1:
            param = float(param)
            continue
        param = int(param)
    return act_funct_dict[all_params[0]](*all_params[1:])