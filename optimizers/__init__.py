from .Optimizer import Optimizer
from .StandardGD import StandardGD
from .Momentum import Momentum
from .Nesterov import Nesterov
from .RMSProp import RMSProp
from .ADAGrad import ADAGrad
from .ADAM import ADAM

from schedulers import parse_sched_info

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
    
    for param in all_params[1:-1]:
        if param.find('.') != -1:
            param = float(param)
            continue
        param = int(param)
    sched = parse_sched_info(all_params[-1])
    return opt_dict[all_params[0]](*all_params[1:-1], sched)