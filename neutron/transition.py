from typing import Any, DefaultDict, List
import random

class Transition:
    """For defining the logic of a transition between nodes.
    We must keep track of the mode we need to transition to
    """
    def __init__(self, mode1, mode2, nn=None):
        self.mode1 = mode1
        self.mode2 = mode2
        self.nn = nn

    def __call__(self) -> Any:
        """Check the guard condition(s) + return the new mode and state"""
        # write a wrapper to check that this __call__ override meets the
        # requirements (i.e. updates all state variables)

        # this function needs to update the mode and state variables
        print('Not implemented.')
        return
        

class TS:
    # TODO: Implement this like State
    def __init__(self, ts):
        self.ts = ts
