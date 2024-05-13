# PSL imports

# third-party imports
import pytest

# local imports
from neutron.automata import Neutron
from neutron.mode import Mode, MS
from neutron.state import State, SS
from neutron.transition import Transition, TS

if __name__ == "__main__":
    print('Starting testing...')
    print('\n')
    
    print('----- MODE testing -----')
    mode1 = Mode(name='Mode 1', nn=None, initial=True)
    mode2 = Mode(name='Mode 2', nn=None)
    
    modes = [mode1, mode2]
    ms = MS(modes)

    assert(mode1.name == 'Mode 1')
    print('Test #1 passed.')
    assert(mode2.name == 'Mode 2')
    print('Test #2 passed.')
    print('\n')

    print('----- TRANSITION testing -----')
    t1 = Transition(mode1, mode2)
    t2 = Transition(mode2, mode1)
    t3 = Transition(mode1, mode1)
    t4 = Transition(mode2, mode2)

    transitions = [t1, t2, t3, t4]
    ts = TS(transitions)

    print('Succesfully created transitions.')
    print('\n')

    print('----- NEUTRON testing -----')
    nn = None
    N = Neutron(nn, ms, ts)

#    N.add_mode(mode1)
#    N.add_mode(mode2)

#    N.addT(t1)
#    N.addT(t2)
#    N.addT(t3)
#    N.addT(t4)

 

    print('Successfully created Neutron instance.')
    print('\n')
    print('----- NEUTRON summary -----')
    print(N)
    print('\n')

    print('Yay!')
