from .PoolFunc import PoolFunc
from .Average import Average
from .Minimum import Minimum
from .Maximum import Maximum

pool_funct_dict: dict[str, PoolFunc] = {"Maximum":Maximum, 
                                        "Average":Average, 
                                        "Minimum":Minimum}

def parse_pool_info(input: str) -> PoolFunc:
    all_params = input.strip().split("/")
    if all_params[0] not in pool_funct_dict:
        return PoolFunc()
    
    for param in all_params[1:]:
        if param.find('.') != -1:
            param = float(param)
            continue
        param = int(param)
    return pool_funct_dict[all_params[0]](*all_params[1:])