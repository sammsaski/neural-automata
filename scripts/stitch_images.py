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


def stitch_images(image1, image2):
    """Put two images side-by-side."""
    return np.hstack((image1, image2))


def create_expression(digits, operators, nd1, nd2, op='random'):
    """
    Create arithmetic expression by stitching images from the Handwritten Digits and Operators dataset.

    digits: set of images of digits to sample from
    operators: set of images of operators to sample from
    nd1: number of digits for the first operand
    nd2: number of digits for the second operand
    op: the arithmetic operation to perform
    """
    # # load the data
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

    digit_labels = ['0', '1', '2', '3', '4', '5' ,'6', '7', '8', '9']
    op_labels = ['+', '-', '*', '%']
    
    digits = data[np.isin(data[:, 1], digit_labels)]
    operators = data[np.isin(data[:, 1], op_labels)]

    create_expression(digits, operators, nd1=2, nd2=2)