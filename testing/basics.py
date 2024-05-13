# PSL imports

# third-party imports

# local imports
from neutron.automata import Neutron
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, TS

if __name__ == "__main__":
    print('Starting testing...')
    print('\n')
    
    print('----- MODE testing -----')
    mode1 = Mode(name='Digits', nn=None, initial=True)
    mode2 = Mode(name='Letters', nn=None)
    
    modes = [mode1, mode2]
    ms = MS(modes)

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


    print('Successfully created Neutron instance.')
    print('\n')
    print('----- NEUTRON summary -----')
    print(N)
    print('\n')

    print('Yay!')
