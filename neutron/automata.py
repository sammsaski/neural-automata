from re import error
from typing import Any, DefaultDict, List
import random

from neutron.mode import Mode, MS
from neutron.state import State 
from neutron.transition import Transition, TS

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
        self.S = S
        self.directed = directed
        self.nn = nn # neural controller

        if self.M:
            self.I = [m for m in self.M.modes if m.initial] # TODO: change T to inherit from dictionary
            if len(self.I) == 0: 
                raise error('No initial states defined.')

            if len(self.I) == 1:
                self.current_mode= self.I[0]
            else:
                self.current_mode= random.choice(self.I)
    
    def move(self, x) -> Transition:
        """Logic for deciding which transition to take.""" 
        # 0 = digit
        # 1 = letter

        res = self.nn(x)

        # this should return the transition to take
        # TODO: make sure transitions have UIDs so that we can refer to them safely
        if self.current_mode == 0 and res == 1:
            return 
        elif self.current_mode == 0 and res == 0:
            return
        elif self.current_mode == 1 and res == 0:
            return
        else:
            # self.current_mode == 1 and res == 1
            return

    def step(self, x):
        """At every step, we take an input decide which NN to use with the
        neural controller, and then perform the desired task."""
        
        # determine which transition to take 
        t = self.move(x)

        # infer
        res = self.nn(x)

        # update state variables after inference
        self.state_vars['x'] += 1
        self.state_vars['y'] += 1

    # ------------------------------------------------ #
    # ------------------ MODE METHODS ---------------- #
    # ------------------------------------------------ #
    def add_mode(self, new_mode) -> None:
        if self.M:
            self.M.append(new_mode)
            self._update_num_modes()
            return

    def build_edge(self, mode1: Mode, mode2: Mode, t: Transition) -> None:
        self.edges[(mode2, mode2)] = t
        self._update_num_edges()
        return

    def check_neutron(self,):
        pass

    def _update_num_modes(self,):
        self.num_modes= len(self.M) 

    def _update_num_edges(self,):
        self.num_edges = len(self.edges) 

    # ------------------------------------------------ #
    # --------------- TRANSITION METHODS ------------- #
    # ------------------------------------------------ #
    def createT(self, mode1, mode2):
        pass

    def addT(self, transition):
        if self.T:
            self.T.append(transition)

    def removeT(self, transition):
        if self.T:
            self.T.remove(transition)

    def __str__(self,):
        return ''.join([f'{t.mode1.name} --> {t.mode2.name}\n' for t in self.T.ts])

if __name__ == "__main__":
   pass 
