import neutron.automata as N
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, 15:51:16

def m1m2(N, T):
    pass

def m2m1(N, T):
    pass

def m1m1(N, T):
    pass

def m2m2(N, T):
    pass

class SimpleMathAutomaton(N.Neutron):

    def move(self, x) -> Transition:
        # self.nn is binary classifier between digits and operators
        if self.nn:
            # 0 = digit, 1 = operator
            res = self.nn(x)
            
            if self.current_mode == self.M.Digits and res == 0:
                return self.m1m1
            elif self.current_mode == self.M.Digits and res == 1:
                return self.m1m2
            elif self.current_mode == self.M.Operators and res == 0:
                return self.m2m1
            else:
                # self.current_mode == self.M.Operators and res == 1
                return self.m2m2

if __name__ == "__main__":
    # Modes
    mode1 = Mode(name='Digits', nn=None, initial=True)
    mode2 = Mode(name='Operators', nn=None)
    modes = [mode1, mode2]
    ms = MS(modes)

    # States
    ss = State({'first': [], 'second': [], 'op': None})

    # Transitions
    t1 = Transition(mode1, mode2, m1m2, id='m1m2')
    t2 = Transition(mode2, mode1, m2m1, id='m2m1')
    t3 = Transition(mode1, mode1, m1m1, id='m1m1')
    t4 = Transition(mode2, mode2, m2m2, id='m2m2')
    transitions = [t1, t2, t3, t4]
    ts = TS(transitions)

    # Neutron
    nn = None
    N = SimpleMathAutomaton(nn, ms, ts, ss)

    # Use the automaton
    def use_op(c, in1, in2):
        if c == '+':
            return in1 + in2
        elif c == '-':
            return in1 - in2
        elif c == '*':
            return in1 * in2
        else:
            # c == '%'
            return in1 / in2

    print(f'Mode: {N.current_mode}, State: ' + ', '.join([f'{attr}={value}' for attr, value in N.S.__dict__.items()]))
    inputs = []

    for inp in inputs:

        for step in range(len(inp)):
            # Get the input
            in = inp[step]

            N.step(in)

            output = f'Mode: {N.current_mode}, State: ' + ', '.join([f'{attr}={value}' for attr, value in N.S.__dict__.items()]) + ' ' + f'Input: {in}'
            print(output)

        # Sequence complete, get the output
        in1 = int(''.join(N.S.first))
        in2 = int(''.join(N.S.second))
        op = N.S.op
        res = use_op(op, in1, in2)
        print(res)







