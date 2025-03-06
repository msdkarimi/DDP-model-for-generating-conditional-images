from inspect import isfunction

def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d