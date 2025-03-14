
from collections.abc import Callable
from ml_collections import ConfigDict
import inspect
import os


__all__ = ['register_model', 'builder']

_registry = dict()

def register_model(fun:Callable):
    # module_name = fun.__file__.split('.')[-1]
    module_name = inspect.getfile(fun).split(os.sep)[-1].split('.')[0]
    if module_name not in _registry:
        _registry[module_name] = fun
    else:
        print(f'\t -->{module_name} has been registered earlier!', )

    print(f'OK-{module_name} constructor registered!', )
    return fun

def builder(name, *args, **configs):
    if name not in _registry:
        print(f'\t --{name} constructor not registered yet!')
        return None
    else:
        print(f'OK-built model {name} from config!')
        return _registry[name](*args, **configs)
