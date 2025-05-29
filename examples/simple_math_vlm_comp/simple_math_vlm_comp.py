# standard lib
import base64
import operator
import os
import random
import sys
import time
import signal

# third-party
import torch
import numpy as np
import ollama
from PIL import Image

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

from examples.simple_math_vlm_comp.setup_experiment import evaluate_expression
from examples.simple_math_vlm_comp.networks.cnn import CNN

DIGIT_CHOICES = [str(i) for i in range(10)]
OPERATOR_CHOICES = ['+', '-', '%', '*']
OP_MAP = {10: '+', 11: '-', 12: '*', 13: '%'}
OPS = {'+': operator.add, '-': operator.sub, '*': operator.mul, '%': operator.truediv}

timed_out = False

def alarm_handler(signum, frame):
    global timed_out
    timed_out = True
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, alarm_handler)


# TODO: Rename to neurosymbolic_automaton
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

    start = time.time()

    # use the automaton to predict the arithmetic expression
    for element in arithmetic_expression:
        type_logits = nn(element)
        type_prob = torch.sigmoid(type_logits)
        type_pred = (type_prob >= 0.5).float()

        # digit classification
        if type_pred == 0: 
            logits = digit_nn(element)
            pred = torch.argmax(logits)
            pred = str(int(pred))

        # operator classification
        else:
            logits = operator_nn(element)
            pred = torch.argmax(logits)
            pred = OP_MAP[pred.item()]
        
        predicted_expression.append(pred)

    # solve the expression
    res = evaluate_expression(predicted_expression)

    end = time.time()

    return res, end - start

def neural_automaton2(arithmetic_expression):
    """
    Given an arithmetic expression (as a list of images describing it), return the output of the arithmetic expression
    described by the operands and operators.
    """
    base_fp = os.path.join(os.getcwd(), 'examples', 'simple_math_vlm_comp')

    # load the models
    model_fp = os.path.join(base_fp, 'models', 'torch', 'HDO.pth')
    
    # Mode Switch Model
    nn = CNN()
    nn.load_state_dict(torch.load(model_fp))

    predicted_expression = []

    start = time.time()

    # use the automaton to predict the arithmetic expression
    for element in arithmetic_expression:
        logits = nn(element)
        pred = torch.argmax(logits)

        # operator
        if pred > 9:
            pred = OP_MAP[pred.item()]
        else:
            pred = str(int(pred))

        predicted_expression.append(pred)

    # solve the expression
    res = evaluate_expression(predicted_expression)

    end = time.time()

    return res, end - start



def get_na_output(num_samples, num_operands):
    data_fp = 'examples/simple_math_vlm_comp/data'
    na_fp = os.path.join(data_fp, 'na', str(num_operands))
    # results_fp = os.path.join('examples/simple_math_vlm_comp/results/na', str(num_operands))
    results_fp = os.path.join('examples/simple_math_vlm_comp/results/na2', str(num_operands))
    labels_fp = os.path.join(data_fp, f'{num_operands}_labels.txt')

    # create directory
    if not os.path.isdir(results_fp):
        os.mkdir(results_fp)

    samples = []

    with open(labels_fp, 'r') as f:
        for sample_num in range(num_samples):
            sample_fp = next((fp for fp in os.listdir(na_fp) if fp.startswith(f'sample_{sample_num}_')), None)
            sample_fp = os.path.join(na_fp, sample_fp)
            line = f.readline()
            samples.append([sample_fp] + [sample.strip() for sample in line.split(',')])

    results_file = os.path.join(results_fp, f'results.txt')

    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(samples):
        # read the images and construct the sample list of images
        sample_data = np.load(sample_fp)
        arithmetic_expression = [torch.from_numpy(sample_data[i]).unsqueeze(0).float() for i in range(sample_data.shape[0])]
        # res, time_taken = neural_automaton(arithmetic_expression)
        res, time_taken = neural_automaton2(arithmetic_expression)

        with open(results_file, 'a+') as f:
            # res, time_taken = neural_automaton(arithmetic_expression)
            res, time_taken = neural_automaton2(arithmetic_expression)
            print(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}')
            f.write(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}\n')


def vlm(model_str, sample_fp, true_expression):
    """
    Given an arithmetic expression (as an image), return the output of the arithmetic expression
    described by the operands and operators.

    arithmetic expression (list): a png representing the arithmetic expression
    """
    # with open(sample_fp, "rb") as image:
        # encoded_image = base64.b64encode(image.read()).decode("utf-8")
    
    # task 1 : correctly classifying the arithmetic expression and evaluating it
    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                'role': 'user',
                'content': 'Solve the mathematical expression in the image. The output must be in the format <numerical expression in image>=<solution>. DO NOT give me any other output. Also, DO NOT use LaTeX. These are simple expressions and can be expressed simply.',
                # 'images': [encoded_image]
                'images': [sample_fp]
            }
        ]
    )
    end = time.time()

    return res['message']['content'], end - start


def vlm_sequence(model_str, sample_fp):
    """
    Given an arithmetic expression (as a list of images describing it), return the output of the arithmetic expression
    described by the operands and operators.

    model_str (str): the name of the VLM to use
    sample_fps (str): filepath of the directory containing the images (pngs) to give as input
    """
    image_fps = [os.path.join(sample_fp, image_fp) for image_fp in os.listdir(sample_fp)]

    input_prompt = '''
        You will be provided a sequence of input images. Contained in each image will be
        either a digit (0-9) or an operator (+, -, /, *). Read these together to make an
        arithmetic expression. You need to solve these left to right, i.e. keep a running
        total of the value as you read in each image to make valid arithmetic expressions. For
        example, given a sequence of images like ['5', '+', '1', '*', '2], you would first
        read the valid expression '5+1' and evaluate it to 6. Then, you would read the
        next operator and operand to get the valid expression '6*2', which is evaluated
        to 12.

        The output must be in the format <numerical expression in image>=<solution>. 
        DO NOT give me any other output. Also, DO NOT use LaTeX. These are simple expressions 
        and can be expressed without the use of LaTeX.'
    '''

    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                'role': 'user',
                'content': input_prompt,
                'images': image_fps
            }
        ]
    )
    end = time.time()

    return res['message']['content'], end - start


def get_vlm_output(model_str):
    """
    Given a set of arithmetic expressions, evaluate them with the neural automaton
    and the VLM and compare the results.

    model_str (str): the name of the VLM model to use
    na_input (list): a list of images (as numpy arrays) describing the arithmetic expression to evaluate
    vlm_input (list of png(s)): one or several pngs used to describe the arithmetic expression we want solved by the VLM
    """
    data_fp = 'examples/simple_math_vlm_comp/data'
    vlm_fp = os.path.join(data_fp, 'vlm')
    labels_fp = os.path.join(data_fp, 'labels.txt')

    samples = []

    with open(labels_fp, 'r') as f:
        for sample in os.listdir(vlm_fp):
            sample_fp = os.path.join(vlm_fp, sample)
            line = f.readline()
            samples.append([sample_fp] + [sample.strip() for sample in line.split(',')])

    results_fp = os.path.join(data_fp, f'results_{model_str}.txt')

    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(samples):
        with open(results_fp, 'a') as f:
            res, time_taken = vlm(model_str, sample_fp, true_expression)
            print(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}')
            f.write(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}\n')


def get_vlm_output_long_expression(model_str, num_operands):
    """
    Given a set of long arithmetic expressions (# operands > 2), evaluate them with the neural automaton
    and the VLM and compare the results.

    model_str (str): the name of the VLM model to use
    """
    data_fp = 'examples/simple_math_vlm_comp/data'
    vlm_sequence_fp = os.path.join(data_fp, 'vlm', 'sequence', str(num_operands))
    results_fp = os.path.join('examples/simple_math_vlm_comp/results/sequence', str(num_operands))
    labels_fp = os.path.join(data_fp, f'{num_operands}_labels.txt')

    samples = []

    with open(labels_fp, 'r') as f:
        all_sample_fps = os.listdir(vlm_sequence_fp)
        all_sample_fps.sort()
        for sample in all_sample_fps:
            sample_fp = os.path.join(vlm_sequence_fp, sample) # /path/to/vlm/sequence/{num_operands}/sample_0
            line = f.readline()
            samples.append([sample_fp] + [sample.strip() for sample in line.split(',')])

    results_file = os.path.join(results_fp, f'results_{model_str}.txt')

    for sample_num, (sample_fp, true_expression, true_solution) in enumerate(samples):
        if sample_num > 24:
            continue

        global timed_out
        timed_out = False  # Reset flag at start of each sample

        signal.alarm(180)
        try:
            res, time_taken = vlm_sequence(model_str, sample_fp)
        except TimeoutError:
            signal.alarm(0)
            with open(results_file, 'a') as f:
                print(f'#{sample_num} -> {true_expression}={true_solution} | timeout')
                f.write(f'#{sample_num} -> {true_expression}={true_solution} | timeout: 180\n')
            continue
        except Exception as e:
            signal.alarm(0)
            with open(results_file, 'a') as f:
                print(f'#{sample_num} -> {true_expression}={true_solution} | error: {e}')
                f.write(f'#{sample_num} -> {true_expression}={true_solution} | error: {e}\n')
            continue
        signal.alarm(0)

        with open(results_file, 'a') as f:
            if timed_out:
                print(f'#{sample_num} -> {true_expression}={true_solution} | timeout')
                f.write(f'#{sample_num} -> {true_expression}={true_solution} | timeout: 180\n')
            else:
                print(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}')
                f.write(f'#{sample_num} -> {true_expression}={true_solution} | {res}: {time_taken}\n')


if __name__=="__main__":
    models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']

    # get vlm output
    for model in models:
        get_vlm_output(model)

    for model in models:
        for num_operand in range(3, 11):
            get_vlm_output_long_expression(model, num_operand)

    
    for num_operands in range(2, 11):
        get_na_output(100, num_operands=num_operands)