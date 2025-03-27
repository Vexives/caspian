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
    
    for param in all_params[1:]:
        if param.find('.') != -1:
            param = float(param)
            continue
        try:
            param = int(param)
        except:
            param = None
    return sched_dict[all_params[0]](*all_params[1:])