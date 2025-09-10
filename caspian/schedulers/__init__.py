from .Scheduler import Scheduler
from .LambdaLR import LambdaLR
from .SchedulerLR import SchedulerLR
from .StepLR import StepLR
from .LinearLR import LinearLR
from .ConstantLR import ConstantLR

sched_dict: dict[str, Scheduler] = {"SchedulerLR":SchedulerLR,
                                    "LambdaLR":LambdaLR,
                                    "StepLR":StepLR,
                                    "LinearLR":LinearLR,
                                    "ConstantLR":ConstantLR}

def parse_sched_info(input: str) -> Scheduler:
    all_params = input.strip().split(":")
    if all_params[0] not in sched_dict:
        return Scheduler()
    
    format_params = list(map(__map_to_numeric, all_params[1:]))
    return sched_dict[all_params[0]](*format_params)

def __map_to_numeric(input: str) -> tuple | int:
    if input.find('.') != -1:
        return float(input)
    try:
        return int(input)
    except:
        return None