from typing import Any, DefaultDict, List
import random

from neutron.mode import Mode, MS
from neutron.state import State 
from neutron.transition import Transition, TS


# ! TODO ! -- Refactor this to be abstract methods and build out the original example + new example with mathematical operators

class Neutron:
    """A neural automata."""
    def __init__(self, nn, M: MS, T: TS, S: State = State(), directed=True):
        """The building of every automata will begin by defining the number of modes
        desired.

        An automata is made up of:
        - M : a set of modes
        - S : a set of state variables
        - T : a set of transitions between modes
        - I : a set of initial modes that are a subset of S
        - X : a set of input variables
        - Y : a set of output variables
        """
        self.M = M 
        self.T = T

        # add ref for this automaton to the transitions, may want to change this to be a TS operation instead of individual transitions
        for t in T.__dict__.values():
            t.add_ref(self) 

        self.S = S
        self.directed = directed
        self.nn = nn # neural controller

        # define an initial mode for the automaton
        if self.M:
            self.I = [m for m in self.M.__dict__.values() if m.initial] # TODO: change T to inherit from dictionary
            if len(self.I) == 0: 
                raise error('No initial states defined.')

            if len(self.I) == 1:
                self.current_mode= self.I[0]
            else:
                self.current_mode= random.choice(self.I)

    def move(self, x) -> Transition:
        """Logic for deciding which transition to take.""" 
        raise NotImplementedError('This method needs to be implemented.')

    def step(self, x):
        """At every step, we take an input decide which NN to use with the
        neural controller, and then perform the desired task."""
        
        # determine which transition to take 
        t = self.move(x)

        # take the transition
        # take_input might be unnecessary, we can just pass the args to the network if it needs them i.e. use *args
        if t.take_input:
            t(x)
        else:
            t()

        # infer
        #if self.nn:
        #    res = self.nn(x)

    def check_neutron(self,):
        raise NotImplementedError('This method needs to be implemented.')

    def update_mode(self, m):
        if m not in self.M:
            raise Exception(f'{m} not in the set of modes {self.M}.')

        self.current_mode = m

    def __str__(self,):
        return '\n'.join([f'{t.mode1.name} --> {t.mode2.name}' for t in self.T.__dict__.values()])

if __name__ == "__main__":
   pass 
