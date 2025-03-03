__all__ = ['register_model', 'build_model']

from collections.abc import Callable
from ml_collections import ConfigDict
import inspect
import os
_registry = dict()

__all__ = ['register_model', 'builder']

def register_model(fun:Callable):
    # module_name = fun.__file__.split('.')[-1]
    module_name = inspect.getfile(fun).split(os.sep)[-1].split('.')[0]
    _registry[module_name] = fun
    print(f'{module_name} constructor registered!', )
    return fun

def builder(config:ConfigDict):
    if config.name not in _registry:
        print(f' {config.name} constructor not registered yet!')
        return False
    else:
        print(f'built model {config.name} from config!')
        return _registry[config.name](**config)
