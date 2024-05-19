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


class Timer:
    def __init__(self, label: str = 'timer'):
        self.label = label
        self.__start = time.time()
        self.__last = self.__start
        self.__records = []

    @property
    def current_time(self):
        return time.time() - self.__start
    
    @property
    def last_record(self):
        return self.__last
    
    def step(self):
        t = self.record()
        return t - self.__last
    
    def record(self):
        t = self.current_time
        self.__records.append(t)
        self.__last = t
        return t
    
    @property
    def records(self):
        return tuple(self.__records)
    
    def clear(self):
        self.__last = self.__start
        self.__records.clear()
    
    def reset(self):
        self.__start = time.time()
        self.clear()

    def __enter__(self):
        self.reset()
        return self
    
    def __exit__(self, *args):
        print(f"[{self.label}] exits after {self.current_time} seconds.")
