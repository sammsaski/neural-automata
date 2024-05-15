# PSL imports

# third-party imports

# local imports
from neutron.automata import Neutron
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, TS

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
    print('invoked!!')
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


if __name__ == "__main__":
    print('Starting testing...')
    print('\n')
    
    print('----- MODE testing -----')
    mode1 = Mode(name='Digits', nn=None, initial=True)
    mode2 = Mode(name='Letters', nn=None)
    
    modes = [mode1, mode2]
    ms = MS(modes)

    print('\n')

    print('----- STATE testing -----')
    ss = State({'x': 0, 'y': 0})

    print('----- TRANSITION testing -----')
    t1 = Transition(mode1, mode2, m1m2, id='m1m2')
    t2 = Transition(mode2, mode1, m2m1, id='m2m1')
    t3 = Transition(mode1, mode1, m1m1, id='m1m1')
    t4 = Transition(mode2, mode2, m2m2, id='m2m2')

    transitions = [t1, t2, t3, t4]
    ts = TS(transitions)

    print('Succesfully created transitions.')
    print('\n')

    print('----- NEUTRON testing -----')
    nn = None
    N = Neutron(nn, ms, ts, ss)


    print('Successfully created Neutron instance.')
    print('\n')
    print('----- NEUTRON summary -----')
    print(N)
    
    print(f'Mode: {N.current_mode}, starting state: ' + ', '.join([f'{attr} : {value}' for attr, value in N.S.__dict__.items()]))
    while True:
        x = input('Enter a letter or digit.\n')
        output = f'Mode: {N.current_mode}, ' + ', '.join([f'{attr} : {value}' for attr, value in N.S.__dict__.items()])
        N.step(x)
        output += ' --> '
        output += f'Mode: {N.current_mode}, ' + ', '.join([f'{attr} : {value}' for attr, value in N.S.__dict__.items()])
        output += '\n'
        print(output)
