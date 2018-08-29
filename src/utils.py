import time
import numpy as np
import json
from .logging import logger
# Timing and Performance

def timing_info(method):
    def wrapper(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        logger.info(f"timing_info: {method.__name__}"
                    f"@{round((end_time-start_time)*1000,1)} ms")

        return result

    return wrapper

def record_time_interval(section, start_time, line_break=False):
    """Record a time interval since the last timestamp"""
    end_time = time.time()
    delta = end_time - start_time
    if delta < 1:
        delta *= 1000
        units = "ms"
    else:
        units = "s"
    if line_break:
        logger.info("PROCESS_TIME:{:>36}    {} {}\n".format(section, round(delta, 1), units))
    else:
        logger.info("PROCESS_TIME:{:>36}    {} {}".format(section, round(delta, 1), units))
    return end_time

def normalize_numpy_dict(d):
    ret = d.copy()
    for k, v in ret.items():
        if isinstance(v, np.generic):
            ret[k] = np.asscalar(v)
    return ret

def save_json(filename, obj):
    with open(filename, 'w') as fw:
        json.dump(obj, fw, indent=2, sort_keys=True)
