import copy
from functools import wraps

def deep_copy_args(func):
    """is a decorator that makes a deep copy of each of the arguments

    Args:
        func (_type_): the function

    Returns:
        _type_: the decorator
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        copied_args = copy.deepcopy(args)
        copied_kwargs = copy.deepcopy(kwargs)
        return func(*copied_args, **copied_kwargs)
    return wrapper
