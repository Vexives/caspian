from .Optimizer import Optimizer
from .StandardGD import StandardGD
from .Momentum import Momentum
from .Nesterov import Nesterov
from .RMSProp import RMSProp
from .ADAGrad import ADAGrad
from .ADAM import ADAM

from ..schedulers import parse_sched_info

opt_dict: dict[str, Optimizer] = {"StandardGD":StandardGD,
                                  "Momentum":Momentum,
                                  "Nesterov":Nesterov,
                                  "RMSProp":RMSProp,
                                  "ADAGrad":ADAGrad,
                                  "ADAM":ADAM}

def parse_opt_info(input: str) -> Optimizer:
    all_params = input.strip().split("/")
    if all_params[0] not in opt_dict:
        return Optimizer()
    
    format_params = list(map(__map_to_numeric, all_params[1:-1]))
    sched = parse_sched_info(all_params[-1])
    return opt_dict[all_params[0]](*format_params, sched)

def __map_to_numeric(input: str) -> tuple | int:
    try:
        return int(input)
    except:
        return float(input)