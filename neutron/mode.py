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
        return f'Mode `{self.name}`'

class MS:
    """The set of modes in the neural automata."""
    def __init__(self, modes):
        self.modes = modes
