import os
from PIL import Image
import numpy as np
import random
import sys

"""
For use with the Handwritten Digits and Operators dataset available at
https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/data

The dataset contains a training, validation, and testing split of the dataset as numpy.ndarrays.

Each sample has an image with dimensions (28, 28) and a label that is the digit or operator in string form.
"""

DIGIT_CHOICES = [str(i) for i in range(10)]
OPERATOR_CHOICES = ['+', '-', '%', '*']

def stitch_images(image1, image2):
    """Put two images side-by-side."""
    return np.hstack((image1, image2))


def generate_operand(fixed_length=None):
    """Generate an operand as a list of digits with a fixed or random length."""
    length = fixed_length if fixed_length else random.randint(1, 5)
    operand = [random.choice(DIGIT_CHOICES) for _ in range(length)]

    # ensure no 0 at start of operand
    while operand[0] == '0':
        operand[0] = random.choice(DIGIT_CHOICES)

    return operand


def generate_expressions():
    expressions = {}

    for num_operands in range(2, 11):
        expressions[num_operands] = []

        # 25 fixed-length operand expressions
        for _ in range(25):
            for operand_length in range(1, 6):
                operands = [generate_operand(fixed_length=operand_length) for _ in range(num_operands)]
                operators = [random.choice(OPERATOR_CHOICES) for _ in range(num_operands-1)]
                
                # expression = sum(([*op, optr] for op, optr in zip(operands, operators)), []) + operands[-1] # interleave operands + operators
                
                expression = [operands, operators]
                expressions[num_operands].append(expression)

        # 25 random-length operand expressions
        for _ in range(25):
            operands = [generate_operand() for _ in range(num_operands)]
            operators = [random.choice(OPERATOR_CHOICES) for _ in range(num_operands-1)]
            
            # expression = sum(([*op, optr] for op, optr in zip(operands, operators)), []) + operands[-1] # interleave operands + operators
            
            expression = [operands, operators]
            expressions[num_operands].append(expression)

    return expressions


def create_expression(digits, operators, target_operands, target_operators, filepath=None):
    """
    Create arithmetic expression by stitching images from the Handwritten Digits and Operators dataset.

    digits: set of images of digits to sample from
    operators: set of images of operators to sample from
    target_operands: a list of lists where each sublist corresponds to an operand (i.e. # of sublists = # of operands) and each element in the list is the target digit.
    target_operators: a list of target operators, should have length = len(target_digits)-1
    """
    # load the data
    # data = np.load(fp)
    # images = data[:, 0] # training images have 200331 samples that are (28, 28)
    
    # get the images to be used for the operators
    filtered_operators = [random.choice(operators[operators[:, 1] == op].tolist()) for op in target_operators]

    # get the images to be used for the operands
    filtered_operands = [[random.choice(digits[digits[:, 1] == digit].tolist()) for digit in operand] for operand in target_operands]

    # separate the images + labels
    operand_images = [[image for image, label in operand] for operand in filtered_operands]
    operand_labels = [[label for image, label in operand] for operand in filtered_operands] 
    
    operator_images = [image for image, label in filtered_operators]
    operator_labels = [label for image, label in filtered_operators]

    # build the expression
    expression = []
    expression_labels = []

    for i in range(len(operand_images)):
        expression.extend(operand_images[i])
        expression_labels.extend(operand_labels[i])

        if i < len(operator_images):
            expression.append(operator_images[i])
            expression_labels.append(operator_labels[i])

    # join the images together horizontally
    np_expression = np.hstack(expression)

    # save the image
    im = Image.fromarray((np_expression * 255).astype(np.uint8))
    im_path = os.path.join(filepath if filepath else '', f"test_exp={''.join(expression_labels)}.png")
    im.save(im_path)


def create_expression_random(digits, operators, nd1, nd2, op='random'):
    """
    Create arithmetic expression by stitching images from the Handwritten Digits and Operators dataset.

    digits: set of images of digits to sample from
    operators: set of images of operators to sample from
    nd1: number of digits for the first operand
    nd2: number of digits for the second operand
    op: the arithmetic operation to perform
    """
    # load the data
    # data = np.load(fp)
    # images = data[:, 0] # training images have 200331 samples that are (28, 28)
    digits_list = digits.tolist()

    operand1 = list(random.sample(digits_list, nd1))
    operand2 = list(random.sample(digits_list, nd2))
    
    if op == 'random':
        operators_list = operators.tolist()
        operator = random.sample(operators_list, 1)[0] # unpack
    else:
        # filter for the chosen operator and sample
        if op not in ['+', '-', '*', '%']:
            raise ValueError("Invalid operator. Must be one of ['+', '-', '*', '%']")
        
        filtered_operators = operators[operators[:, 1] == op]
        filtered_operators_list = filtered_operators.tolist()
        operator = random.choice(filtered_operators_list)

    # separate images + labels
    operand1_images, operand1_labels = map(list, zip(*operand1))
    operand2_images, operand2_labels = map(list, zip(*operand2))
    operator_images, operator_labels = operator

    np_expression = np.hstack((*operand1_images, operator_images, *operand2_images))

    # save the image
    im = Image.fromarray((np_expression * 255).astype(np.uint8))
    im.save("test.jpeg")


if __name__=="__main__":
    data = np.load('data/handwritten-digits-and-operators/training.npy', allow_pickle=True)
    images = data[:, 0]

    # digit_labels = ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9']
    # op_labels = ['+', '-', '*', '%']
    
    digits = data[np.isin(data[:, 1], DIGIT_CHOICES)]
    operators = data[np.isin(data[:, 1], OPERATOR_CHOICES)]

    # create_expression_random(digits, operators, nd1=2, nd2=2)

    ex = generate_expressions()

    for i in range(2, 11):
        i_expressions = ex[i]

        for j in range(len(i_expressions)):
            target_operands, target_operators = i_expressions[j]
            fp = os.path.join('examples', 'simple_math_vlm_comp', 'data', str(i)) # save to data directory corresponding to num of operands
            create_expression(digits, operators, target_operands=target_operands, target_operators=target_operators, filepath=fp)

        # print(f"Expressions with num_operands={i}")
        # print(f"-"*50)
        # for j in range(len(i_expressions)):
        #     print(''.join(i_expressions[j]))
        # print("\n\n\n")


    # create_expression(digits, operators, target_operands=[['1', '3'], ['4', '5', '6'], ['7', '8', '9']], target_operators=['+', '-'])