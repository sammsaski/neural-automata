import neutron.automata as N
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, TS

# Transition functions
# TODO: Refactor transitions, so that there are no longer individual transitions
#       and instead the transitions are modeled by functions in the TS.

def m1m2(N, T):
    """N is a reference to the automaton."""
    states = N.S
    states.y = max(states.x, states.y)
    states.x = 1
    N.current_mode = T.mode2 
    return

def m2m1(N, T):
    states = N.S
    states.y = max(states.x, states.y)
    states.x = 1
    N.current_mode = T.mode2
    return

def m1m1(N, T):
    states = N.S
    states.y = max(states.x, states.y)
    states.x += 1
    N.current_mode = T.mode2
    return

def m2m2(N, T):
    states = N.S
    states.y = max(states.x, states.y)
    states.x += 1
    N.current_mode = T.mode2
    return

class LongestStreakAutomaton(N.Neutron):
    
    def move(self, x) -> Transition:
        if self.nn:
            res = self.nn(x)
        else:
            res = 0 if x.isnumeric() else 1

        if self.current_mode == self.M.Digits and res == 1:
            return self.T.m1m2
        elif self.current_mode == self.M.Digits and res == 0:
            return self.T.m1m1
        elif self.current_mode == self.M.Letters and res == 0:
            return self.T.m2m1
        else:
            # self.current_mode == self.M.Letters and res == 1
            return self.T.m2m2


if __name__ == "__main__":
    # Modes
    mode1 = Mode(name='Digits', nn=None, initial=True)
    mode2 = Mode(name='Letters', nn=None)
    modes = [mode1, mode2]
    ms = MS(modes)

    # States
    ss = State({'x': 0, 'y': 0})

    # Transitions
    t1 = Transition(mode1, mode2, m1m2, id='m1m2')
    t2 = Transition(mode2, mode1, m2m1, id='m2m1')
    t3 = Transition(mode1, mode1, m1m1, id='m1m1')
    t4 = Transition(mode2, mode2, m2m2, id='m2m2')
    transitions = [t1, t2, t3, t4]
    ts = TS(transitions)

    # Neutron
    nn = None
    N = LongestStreakAutomaton(nn, ms, ts, ss)

    # Use the automaton
    print(f'Mode: {N.current_mode}, starting state: ' + ', '.join([f'{attr} : {value}' for attr, value in N.S.__dict__.items()]))
    while True:
        x = input('Enter a letter or digit.\n')
        output = f'Mode: {N.current_mode}, ' + ', '.join([f'{attr} : {value}' for attr, value in N.S.__dict__.items()])
        N.step(x)
        output += ' --> '
        output += f'Mode: {N.current_mode}, ' + ', '.join([f'{attr} : {value}' for attr, value in N.S.__dict__.items()])
        output += '\n'
        print(output)
