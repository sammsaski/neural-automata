from typing import Any, DefaultDict, List
import random

class State:
    """A state variable in the neural automata."""
    def __init__(self, statevars=None, *args):
        if type(statevars) == list:
            if len(statevars) == len(args):
                for name, value in zip(statevars, args):
                    setattr(self, name, value)
            else:
                raise Exception(f'Incompatible number of state variables {len(statevars)} and initial state values {len(args)}')

        elif type(statevars) == dict:
            for name, value in statevars.items():
                setattr(self, name, value)

        elif type(statevars) == type(None):
            pass

        else:
            raise Exception(f'Unsupported data type {type(statevars)}.')

    def add_state(self, statevars, *args):
        if type(statevars) == list:
            if len(statevars) == len(args):
                for name, value in zip(statevars, args):
                    setattr(self, name, value)

        elif type(statevars) == dict:
            for name, value in statevars.items():
                setattr(self, name, value)

        else:
            # if only one state variable is passed
            setattr(self, statevars, args[0])
    
    def __str__(self) -> str:
        return '\n'.join([f'{attr} : {value}' for attr, value in self.__dict__.items()])
