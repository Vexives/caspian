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
    
    format_params = list(map(__map_to_numeric, all_params[1:]))
    return pool_funct_dict[all_params[0]](*format_params)

def __map_to_numeric(input: str) -> tuple | int:
    try:
        return int(input)
    except:
        return float(input)