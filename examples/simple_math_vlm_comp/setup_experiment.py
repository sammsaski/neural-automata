# standard lib
import operator
import os
import random
import sys
import time

import numpy as np
from PIL import Image


DIGIT_CHOICES = [str(i) for i in range(10)]
OPERATOR_CHOICES = ['+', '-', '%', '*']
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '%': operator.truediv}


"""
Setting up the experiments for the neural automaton paper.

1. Generate the arithmetic expressions that we want to use.
2. For each arithmetic expression:
    i.   Get the images (save as nparray for NA and as stitched image for VLM)
    ii.  Get the label (record the actual arithmetic expression in a .txt file)
    iii. Get the true value of the expression (record the true solution in a .txt file)
"""


def generate_operand(fixed_length=None):
    """Generate an operand as a list of digits with a fixed or random length."""
    length = fixed_length if fixed_length else random.randint(1, 5)
    operand = [random.choice(DIGIT_CHOICES) for _ in range(length)]

    # ensure no 0 at start of operand
    while operand[0] == '0':
        operand[0] = random.choice(DIGIT_CHOICES)

    return operand


def evaluate_expression(arithmetic_expression):
    """
    Given an arithmetic expression (as a list of characters describing it), return the output of the
    arithmetic expression solving from left to right.
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


def generate_expressions(digits_data, operators_data, filepath, num_expressions=25):
    """
    Generate the various arithmetic expressions that we use. Overall, we create an np array of the samples
    (by number of operands), save all stitched together images for the VLMs, and save the actual expression
    label and the true solution when evaluating the expression.

    Format: [ae_1, ae_2, ...] where each ae itself is a list containing sublists that are the operands and characters that are the operators, e.g.
            ae_1 = [['5', '4', '2'], '+', ['1', '2']]

    digits:
    operators:
    filepath: /path/to/data
    labels_filepath:
    """
    expressions = {}

    for num_operands in range(2, 11):
        expressions[num_operands] = []

        # 25 fixed-length operand expressions
        for _ in range(num_expressions):
            for operand_length in range(1, 6):
                operands = [generate_operand(fixed_length=operand_length) for _ in range(num_operands)]
                operators = [random.choice(OPERATOR_CHOICES) for _ in range(num_operands-1)]
                                
                expression = [operands, operators]
                expressions[num_operands].append(expression)

        # 25 random-length operand expressions
        for _ in range(num_expressions):
            operands = [generate_operand() for _ in range(num_operands)]
            operators = [random.choice(OPERATOR_CHOICES) for _ in range(num_operands-1)]
                        
            expression = [operands, operators]
            expressions[num_operands].append(expression)

    # now that we've created the expressions, create the filepaths used for storing the samples and their
    # labels appropriately
    im_base_path = os.path.join(filepath, 'vlm')
    np_base_path = os.path.join(filepath, 'na')
    labels_filepath = os.path.join(filepath, 'labels.txt')

    # organize the arithmetic expressions into valid arithmetic expressions using the lists of operands and operators
    for key in expressions.keys():
        # np array of all the samples
        np_samples = []

        for sample_num, (target_operands, target_operators) in enumerate(expressions[key]):
            # get the images to be used for the operators
            filtered_operators = [random.choice(operators_data[operators_data[:, 1] == op].tolist()) for op in target_operators]

            # get the images to be used for the operands
            filtered_operands = [[random.choice(digits_data[digits_data[:, 1] == digit].tolist()) for digit in operand] for operand in target_operands]

            # separate the images + labels
            operand_images = [[image for image, label in operand] for operand in filtered_operands]
            operand_labels = [[label for image, label in operand] for operand in filtered_operands] 
            
            operator_images = [image for image, label in filtered_operators]
            operator_labels = [label for image, label in filtered_operators]

            # build the expression
            expression = []
            expression_labels = []

            # join the lists of operand images and operator images together to make a coherent arithmetic expression
            for i in range(len(operand_images)):
                expression.extend(operand_images[i])
                expression_labels.extend(operand_labels[i])

                if i < len(operator_images):
                    expression.append(operator_images[i])
                    expression_labels.append(operator_labels[i])

            # join the images together horizontally
            np_expression = np.hstack(expression)
            
            # add the np arrays of the images in the expression to save
            np_samples.append(expression)

            # evaluate the expression
            evaluated_expression = evaluate_expression(expression_labels)

            # add the expression and its value to the labels file
            if labels_filepath:
                with open(labels_filepath, 'w') as f:
                    print(f'{"".join(expression_labels)}, {evaluated_expression}', file=f)

            # save the image
            im_path = os.path.join(im_base_path, str(len(target_operands)), f"sample_{sample_num}__{''.join(expression_labels)}.png")
            im = Image.fromarray((np_expression * 255).astype(np.uint8))
            im.save(im_path)

        # save the np array after adding all samples
        np_samples = np.array(np_samples)
        np_path = os.path.join(np_base_path, f'{len(target_operands)}.npy')
        np.save(np_path, np_samples)


def setup_experiment():
    """Create the samples to be used for experiments"""
    data = np.load('data/handwritten-digits-and-operators/training.npy', allow_pickle=True)

    digits = data[np.isin(data[:, 1], DIGIT_CHOICES)]
    operators = data[np.isin(data[:, 1], OPERATOR_CHOICES)]

    generate_expressions(digits, operators, )

    