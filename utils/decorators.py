from typing import Callable
import time


def timed(label: str):
    def decorator(func: Callable):
        def timed_func(*args, **kargs):
            t1 = time.time()
            r = func(*args, **kargs)
            t2 = time.time()
            print(f"<{label}> returns within {t2 - t1} seconds.")
            return r
        return timed_func
    return decorator
