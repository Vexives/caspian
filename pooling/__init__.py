from .PoolFunc import PoolFunc
from .Average import Average
from .Minimum import Minimum
from .Maximum import Maximum

pool_funct_dict: dict[str, PoolFunc] = {"Maximum":Maximum, 
                                        "Average":Average, 
                                        "Minimum":Minimum}