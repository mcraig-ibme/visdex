import time
import logging
import pandas as pd
from functools import wraps

timing_dict = dict()
timers = dict()


logging.getLogger(__name__)


def timing(f):
    """
    Decorator for timing of full functions.
    :param f:
    :return:
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logging.info(f"#### func:{f.__name__} took: {te-ts:.2f} sec")
        timing_dict[f.__name__] = te - ts
        return result

    return wrap


def start_timer(i):
    """
    Adds timer i to timing_dict and records start time
    """
    timers[i] = time.time()


def log_timing(i, label, restart=True):
    """
    Logs the elapsed time since the timer i was last (re)started, in the timing_dict under
    the supplied label, and restarts the timer unless restart=False.

    This is for use _within_ functions.

    :param i: dictionary key to refer to timer
    :param label: human-readable string to identify the timer
    :param restart: boolean expressing whether the timer should be restarted (if False,
      it is instead removed from timers).
    """
    if i in timers:
        timing_dict[label] = time.time() - timers[i]
        if restart:
            start_timer(i)
        else:
            del timers[i]
    else:
        logging.info(f"Timer {i} attempted to log value but had not been started previously.")


def print_timings():
    """
    Prints timing_dict in a nice format to the logger
    """
    logging.info(f"{pd.DataFrame(timing_dict.items())}")