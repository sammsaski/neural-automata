from typing import Any, DefaultDict, List
import random

class Mode:
    """A mode in the neural automata."""
    def __init__(self, name: str, nn):
        self.name = name
        self.nn = nn

    def infer(self, x):
        return self.nn(x)

    def __str__(self,):
        return f'Mode `{self.name}`'

class Transition:
    """For defining the logic of a transition between nodes.
    We must keep track of the mode we need to transition to
    """
    def __init__(self, nn):
        self.nn = nn

    def __call__(self, input,) -> Any:
        return self.nn(input)

class Neutron:
    """A neural automata."""
    def __init__(self, modes: List, directed=True):
        """The building of every automata will begin by defining the number of modes
        desired.

        An automata is made up of:
        - M : a set of modes
        - S : a set of state variables
        - R : a set of transitions between modes
        - I : a set of initial modes that are a subset of S
        - X : a set of input variables
        - O : a set of output variables
        """
        self.modes = modes 
        self.edges = {}

        self.state_vars = DefaultDict()

        self.num_modes = len(self.modes)
        self.num_edges = len(self.edges)

        self.directed = directed

        self.initial_modes= []
        if len(self.initial_modes) == 1:
            self.current_mode= self.initial_modes[0]
        else:
            self.current_mode= random.choice(self.initial_modes)
    
    def move(self,):
        pass

    def add_mode(self, new_mode) -> None:
        self.modes.append(new_mode)
        self._update_num_modes()
        return

    def build_edge(self, mode1: Mode, mode2: Mode, t: Transition) -> None:
        self.edges[(mode2, mode2)] = t
        self._update_num_edges()
        return

    def check_neutron(self,):
        pass

    def _update_num_modes(self,):
        self.num_modes= len(self.modes) 

    def _update_num_edges(self,):
        self.num_edges = len(self.edges) 

    def __str__(self,):
        return ''.join([f'{mode1.name} --> {mode2.name}\n' for mode1, mode2 in self.edges.keys()])

if __name__ == "__main__":
    # build modes 
    mode1 = Mode(name='Mode 1', nn=None) # leave None for now
    mode2 = Mode(name='Mode 2', nn=None)

    print(mode1)
    print(mode2)
