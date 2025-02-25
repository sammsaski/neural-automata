# standard lib
import operator
import os
import random
import sys
import time

# third-party
import torch
import numpy as np
import ollama

# local
import neutron.automata as N
from neutron.mode import Mode, MS
from neutron.state import State
from neutron.transition import Transition, TS

# network architectures
from examples.simple_math.networks.digit import DigitCNN
from examples.simple_math.networks.operator import OperatorCNN
from examples.simple_math.networks.modeswitch import ModeSwitchCNN

# arithmetic example generation
from scripts.stitch_images import generate_expressions, create_expression

DIGIT_CHOICES = [str(i) for i in range(10)]
OPERATOR_CHOICES = ['+', '-', '%', '*']
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '%': operator.truediv}


def evaluate_expression(arithmetic_expression):
    """
    Given an arithmetic expression (as a list of characters describing it), return the output of the
    arithmetic expression.
    """
    operands = []
    operators = []
    current_operand = []
    
    for item in arithmetic_expression:
        if item in DIGIT_CHOICES:
            current_operand.append(item)
        else:
            operands.append(int(''.join(current_operand))) # current operand over so add it to operands list
            current_operand = [] # reset current operand
            operators.append(item) # store the operator

    if current_operand:
        operands.append(int(''.join(current_operand)))
    
    # edge cases
    if len(operands) != len(operands)-1:
        return -1
    
    if type(operands[0]) != int:
        return -1
    
    if len(operands) < 2 or len(operators < 1):
        return -1

    result = operands[0]
    for i, op in enumerate(operators):
        result = OPS[op](result, operands[i+1]) # apply operator on next operand
    
    return result
    

def neural_automaton(arithmetic_expression):
    """
    Given an arithmetic expression (as a list of images describing it), return the output of the arithmetic expression
    described by the operands and operators.
    """
    base_fp = os.path.join(os.getcwd(), 'examples', 'simple_math')

    # load the models
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

    predicted_expression = []

    # use the automaton to predict the arithmetic expression
    for element in arithmetic_expression:
        element_type = nn(element)
        pred = digit_nn(element) if element_type == 0 else operator_nn(element)
        predicted_expression.append(pred)

    # solve the expression
    res = evaluate_expression(predicted_expression)

    return res

def vlm(model_str, arithmetic_expression):
    """
    Given an arithmetic expression (as a list of images describing it), return the output of the arithmetic expression
    described by the operands and operators.

    arithmetic expression (list): a list of filepaths to pngs representing the image
    """
    images = arithmetic_expression if arithmetic_expression.isinstance(list) else [arithmetic_expression]
    
    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                'role': 'user',
                'content': 'Solve the mathematical operation in the image. DO NOT give me any output besides the numerical value.',
                'images': images
            }
        ]
    )
    end = time.time()

    return res['message']['content'], end - start


def get_output(model_str, na_input, vlm_input):
    """
    Given a set of arithmetic expressions, evaluate them with the neural automaton
    and the VLM and compare the results.

    model_str (str): the name of the VLM model to use
    na_input (list): a list of images (as numpy arrays) describing the arithmetic expression to evaluate
    vlm_input (list of png(s)): one or several pngs used to describe the arithmetic expression we want solved by the VLM
    """
    # get neural automaton output
    neural_automaton_output = neural_automaton(na_input)

    # get VLM output
    vlm_output, vlm_time = vlm()


if __name__=="__main__":
    # 1. ensure the arithmetic expressions have been created
    setup_experiment()

    # 2. get the output of the neural automaton


    # 3. get the output of the VLM
    model = "llava:7b"
    