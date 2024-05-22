from typing import Any, DefaultDict, List
import random

class Mode:
    """A mode in the neural automata."""
    def __init__(self, name: str, nn, initial=False):
        self.name = name
        self.nn = nn
        self.initial = initial

    def __call__(self, x):
        return self.nn(x)

    def __str__(self,):
        return f'{self.name}'

class MS:
    """The set of modes in the neural automata."""
    def __init__(self, modes):
        if type(modes) == list:
            for m in modes:
                setattr(self, m.name, m)
        
        elif type(modes) == dict:
            for name, m in modes.items():
                setattr(self, name, m)

        else:
            raise Exception(f'Unsupported data type for modes: {type(modes)}.')

    def __len__(self):
        return len(self.__dict__)

