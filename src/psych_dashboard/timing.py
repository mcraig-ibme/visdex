import time
import logging
from functools import wraps

timing_dict = dict()

logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        logging.info(f'#### func:{f.__name__} took: {te-ts:.2f} sec')
        timing_dict[f.__name__] = te-ts
        return result
    return wrap
