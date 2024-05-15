from typing import Any, DefaultDict, List
import random
import uuid

class Transition:
    """For defining the logic of a transition between nodes.
    We must keep track of the mode we need to transition to.

    Every object of this class will require a circular ref back to the
    automaton that it is a part of. This way it has access to modes + state
    of the automaton and can manipulate them.
    """
    def __init__(self, mode1, mode2, func, nn=None, id: str = None):
        self.mode1 = mode1
        self.mode2 = mode2
        self.func = func
        self.nn = nn

        # Let user's define their own ID's
        if not id:
            self.id = uuid.uuid4().hex
        else:
            self.id = id

    def __call__(self) -> Any:
        """Check the guard condition(s) + return the new mode and state"""
        # write a wrapper to check that this __call__ override meets the
        # requirements (i.e. updates all state variables)

        # this function needs to update the mode and state variables
        self.func(self.N, self) 

    def add_ref(self, N):
        self.N = N
            

class TS:
    # TODO: Implement this like State
    def __init__(self, ts: List):
        for t in ts:
            if not hasattr(self, t.id):
                setattr(self, t.id, t)
            else:
                raise Exception('ID conflict between transitions. Please check that all transitions have unique IDs.')

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __len__(self):
        return len(self.__dict__)


        

