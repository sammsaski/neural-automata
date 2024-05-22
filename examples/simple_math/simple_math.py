# standard lib
import os
import sys

# third-party
import torch
import numpy as np

# local
import neutron.automata as N
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, TS

# network architectures
from examples.simple_math.networks.digit import DigitCNN
from examples.simple_math.networks.operator import OperatorCNN
from examples.simple_math.networks.modeswitch import ModeSwitchCNN

# NOTE: How does this structure affect performance? How does it affect robustness/generalizability
#       of the model? Is partitioning more efficient? What about for training purposes? How long does
#       training take in comparison?
#      
#       Can we identify mathematical equivalence between AIO models vs. partitioned models into neural automata?
#       I imagine there is some type of linear algebra that we could do to find equivalence.

# define mappings from classes to labels for each nn
MS_MAP = {} # this one maybe not necessary bc only binary classification 
DIGIT_MAP = {}
OP_MAP = {0: '%', 1: '*', 2: '+', 3: '-'}


# TODO: there has to be a better way to define transition functions. What if there are 100 transition functions?
#       You just have to muddle your file with all of their definitions??
# operand -> operator
def m1m2(N, I):
    operands = N.S.Operands
    operators = N.S.Operators

    # change the operand to a real integer
    operands[-1] = int(operands[-1])

    # update current mode
    N.current_mode = N.M.Operators
    # get the classification
    logits = N.current_mode.nn(I)
    pred = int(torch.argmax(logits))
    #pred = OP_MAP.get(pred)
    pred = OP_MAP[pred]
    # add the new operator to the state
    operators.append(pred)

# operator -> operand
def m2m1(N, I):
    operands = N.S.Operands

    # update current mode
    N.current_mode = N.M.Digits

    # get the classification
    logits = N.current_mode.nn(I)
    pred = torch.argmax(logits)
    pred = str(int(pred))

    # start a new operand and add it to the state
    operands.append(pred)

# operand -> operand
def m1m1(N, I):
    operands = N.S.Operands

    # get the classification
    logits = N.current_mode.nn(I)
    pred = torch.argmax(logits)
    pred = str(int(pred)) 
    
    # add the new digit to the end of the current operand
    if not operands:
        operands.append('')
    operands[-1] += pred 

# operator -> operator
def m2m2(N, I):
    operators = N.S.Operators

    # get the classification
    logits = N.current_mode.nn(I)
    pred = int(torch.argmax(logits))
    #pred = OP_MAP.get(pred)
    pred = OP_MAP[pred]
    # add the new operator to the state
    operators.append(pred)

# exit transition
def exitfunc(N):
    def use_op(in1, in2, op):
        if op == '+':
            return in1 + in2
        elif op == '-':
            return in1 - in2
        elif op == '*':
            return in1 * in2
        elif op == '%':
            return in1 / in2
        else:
            raise Exception(f'Unknown operator: {op}')
    
    # verify types of operators
    # TODO: add a helper function to the automata/mode/transition that verifies the types
    #       of the state variables before operating on them
    N.S.Operands = [int(operand) for operand in N.S.Operands]

    operands = N.S.Operands
    operators = N.S.Operators

    if len(operands) - 1 != len(operators): # should be 1 more operand than operator
        raise Exception(f'Incorrect number of operators ({len(operators)}) for number of operands ({len(operands)}).')

    # verify types of operators
    # TODO: add a helper function to the automata/mode/transition that verifies the types
    #       of the state variables before operating on them
    #operands = [int(x) for x in operands]

    op1 = operands.pop(0) # get the first number

    while len(operands) > 0:
        op2 = operands.pop(0) # get the second operand
        operator = operators.pop(0) # get the operator
        op1 = use_op(op1, op2, operator)

    # store the result in the state
    N.S.Result = op1 

    # output the result
    print(f'Mode: {N.current_mode}, State: {N.S.Operands}, {N.S.Operators}, {N.S.Result}')

    # reset the state
    N.S.Result = None


class SimpleMathAutomaton(N.Neutron):

    def move(self, x) -> Transition:
        # define the stopping criteria
        if x == '=':
            return self.T.exit

        # self.nn is binary classifier between digits and operators
        if self.nn:
            # 0 = digit, 1 = operator
            logits = self.nn(x)
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).float()
            
            if self.current_mode == self.M.Digits and pred == 0:
                return self.T.m1m1
            elif self.current_mode == self.M.Digits and pred == 1:
                return self.T.m1m2
            elif self.current_mode == self.M.Operators and pred == 0:
                return self.T.m2m1
            elif self.current_mode == self.M.Operators and pred == 1:
                return self.T.m2m2
            else:
                raise Exception(f'Incomplete automata definition. Missing transition for {self.current_mode} and {pred}.')
           
if __name__ == "__main__":
    """Sample use for the neural automata. This example showcases an automata that
        takes a single image of a digit/operator at each time step and classifies it
        correctly. 

        This can be used to model a system that takes as input images of the digits/operators
        and accurately performs calculations on the incoming stream of data.

        The example cases shown here include:

                5734 + 112 = 5846
                647 - 87   = 560
                31 * 45    = 1395
                120 / 20   = 6.0
    """


    base_fp = os.path.join(os.getcwd(), 'examples', 'simple_math')

    # Load the data
    data_fp = os.path.join(base_fp, 'data')
    data_add = np.load(os.path.join(data_fp, 'add_example.npy'), allow_pickle=True)
    data_sub = np.load(os.path.join(data_fp, 'sub_example.npy'), allow_pickle=True)
    data_mult = np.load(os.path.join(data_fp, 'mult_example.npy'), allow_pickle=True)
    data_div = np.load(os.path.join(data_fp, 'div_example.npy'), allow_pickle=True)

    # Preprocess the data; they are numpy arrays of objects (int matrices, str)
    data_add_samples = [torch.from_numpy(sample).unsqueeze(0).float() for sample, label in data_add]
    data_sub_samples = [torch.from_numpy(sample).unsqueeze(0).float() for sample, label in data_sub]
    data_mult_samples = [torch.from_numpy(sample).unsqueeze(0).float() for sample, label in data_mult]
    data_div_samples = [torch.from_numpy(sample).unsqueeze(0).float() for sample, label in data_div]

    # Models
    model_fp = os.path.join(base_fp, 'models', 'torch') # we are going to use the torch models
    mode_switch_fp = os.path.join(model_fp, 'TrainModeSwitch.pth')
    digit_fp = os.path.join(model_fp, 'TrainDigit.pth')
    operator_fp = os.path.join(model_fp, 'TrainOperator.pth')

    # Mode Switch Model
    nn = ModeSwitchCNN()
    nn.load_state_dict(torch.load(mode_switch_fp))

    # Digit/Operator Models
    digit_nn = DigitCNN()
    digit_nn.load_state_dict(torch.load(digit_fp))

    operator_nn = OperatorCNN()
    operator_nn.load_state_dict(torch.load(operator_fp))

    # Modes
    mode1 = Mode(name='Digits', nn=digit_nn, initial=True)
    mode2 = Mode(name='Operators', nn=operator_nn)
    modes = [mode1, mode2]
    ms = MS(modes)

    # States
    # TODO: naming conventions maybe should be lowercase; uppercase makes it look like a class
    ss = State({'Operands': [], 'Operators': [], 'Result': None})

    # Transitions
    t1 = Transition(mode1, mode2, m1m2, id='m1m2', take_input=True)
    t2 = Transition(mode2, mode1, m2m1, id='m2m1', take_input=True)
    t3 = Transition(mode1, mode1, m1m1, id='m1m1', take_input=True)
    t4 = Transition(mode2, mode2, m2m2, id='m2m2', take_input=True)
    t5 = Transition(None, None, exitfunc, id='exit')
    transitions = [t1, t2, t3, t4, t5]
    ts = TS(transitions)

    # Neutron
    N = SimpleMathAutomaton(nn, ms, ts, ss)

    print(f'Mode: {N.current_mode}, State: ' + ', '.join([f'{attr}={value}' for attr, value in N.S.__dict__.items()]))

        
    print('----- ADDITION EXAMPLE -----')
    while data_add_samples:
        sample = data_add_samples.pop(0)
        N.step(sample)
        print(f'Mode: {N.current_mode}, State: {N.S.Operands}, {N.S.Operators}, {N.S.Result}')
    else:
        N.step('=')
    print('-'*len('----- ADDITION EXAMPLE -----'))
    print('\n')

    print('----- SUBTRACTION EXAMPLE -----')
    while data_sub_samples:
        sample = data_sub_samples.pop(0)
        N.step(sample)
        print(f'Mode: {N.current_mode}, State: {N.S.Operands}, {N.S.Operators}, {N.S.Result}')
    else:
        N.step('=')
    print('-'*len('----- SUBTRACTION EXAMPLE -----'))
    print('\n')

    print('----- MULTIPLICATION EXAMPLE -----')
    while data_mult_samples:
        sample = data_mult_samples.pop(0)
        N.step(sample)
        print(f'Mode: {N.current_mode}, State: {N.S.Operands}, {N.S.Operators}, {N.S.Result}')
    else:
        N.step('=')
    print('-'*len('----- MULTIPLICATION EXAMPLE -----'))
    print('\n')

    print('----- DIVISION EXAMPLE -----')
    while data_div_samples:
        sample = data_div_samples.pop(0)
        N.step(sample)
        print(f'Mode: {N.current_mode}, State: {N.S.Operands}, {N.S.Operators}, {N.S.Result}')
    else:
        N.step('=')
    print('-'*len('----- DIVISION EXAMPLE -----'))

