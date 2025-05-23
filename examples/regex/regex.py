# standard library
import os
import time

# third-party packages
import ollama
import torch
import signal

# local packages
from examples.regex.networks.letter_cnn import LetterCNN
from examples.regex.setup_experiment import matches_regex

# --- timeout handling
TIMEOUT = 100

def handler(signum, frame):
    raise TimeoutError("Function timed out!")

signal.signal(signal.SIGALRM, handler)
signal.alarm(TIMEOUT)

# ---

def neurosymbolic_automaton(input_str, regex):
    """
    Given a string (as images) and a regular language, determine if the
    string should be accepted or not.

    input_str (str): the filepath images being used
    regex (str)
    """
    base_fp = os.path.join(os.getcwd(), 'examples', 'regex')

    # load the data
    images = torch.load(input_str)
    
    # load the model
    model_fp = os.path.join(base_fp, 'models', 'model.pth')
    model = LetterCNN(num_classes=26)
    model.load_state_dict(torch.load(model_fp))

    # predict the input string
    predicted_str_lst = []
    for i in range(images.shape[0]):
        image = images[i]
        outputs = model(image)
        pred = torch.argmax(outputs) # [0, 25]
        pred_c = chr(ord('a') + pred) # convert to character
        predicted_str_lst.append(pred_c)
    
    # join list together to make the string
    predicted_str = ''.join(predicted_str_lst)

    # check if the string matches the regex
    accept = matches_regex(regex, predicted_str)

    return predicted_str, accept


def vlm(model_str, input_str, regex):
    """
    Given a string (as images), a regular language, and the name of
    the VLM to test, prompt the VLM to determine if the string should 
    be accepted or not.

    input_str (str | list): the filepath(s) to the image(s) being used
    """
    if type(input_str) == str:
        p = "an input image"
        input_str = [input_str]
    else: # type(input_str) == list
        p = "a sequence of input images"

    prompt_1 = f'You will be provided {p}. Contained in each image will be a letter of the English alphabet. For this problem we ignore case, so we have only 26 classes. For each image in the sequence, classify it as a letter in the alphabet and then join all of your predictions together to make a string. Remember that only lowercase letters are allowed. You are also given the regular expression {regex}. Now, tell me "accept" if the string that you read should be accepted by the regular expression and "reject" if not. I expect the output in the form <predicted string, accept/reject> and ONLY in this form. You will be marked incorrect if you provide any other outputs or it the output is not in the desired format.'

    # prompt_2 = f'You will be provided a string. Tell me "accept" if the string should be accepted by the regular expression and "reject" if not. I expect ONLY the accept/reject decision. You will be marked incorrect if you provide any other outputs or it the output is not in the desired format.'

    # task 1 : correctly classify the input string and accept/reject it
    start = time.time()
    res = ollama.chat(
        model=model_str,
        messages=[
            {
                'role': 'user',
                'content': prompt_1,
                'images': input_str
            }
        ]
    )
    end = time.time()

    # task 2 : given the regular expression and the string, can it correctly accept/reject it?
    # start2 = time.time()
    # res2 = ollama.chat(
    #     model=model_str,
    #     messages=[
    #         {
    #             'role': 'user',
    #             'content': prompt_2,
    #             'images': input_str
    #         }
    #     ]
    # )
    # end2 = time.time()

    # return res['message']['content'], end - start, res2['message']['content'], end2 - start2
    return res['message']['content'], end - start


def get_na_output():
    """Run all experiments with the neural automaton."""
    na_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', 'na')

    total_str_correct = 0
    total_accept_correct = 0

    for experiment_name in os.listdir(na_fp):
        print(f'working on {experiment_name.split("__")[0]}')
        results_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'results', 'na', f'{experiment_name}.txt')
        labels_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', f'{experiment_name}.txt')
        na_exp_fp = os.path.join(na_fp, experiment_name)

        # get the regex from the experiment name
        regex = experiment_name.split("__")[1]

        with open(labels_fp, 'r') as labels:
            # sort the samples
            samples_fp = os.listdir(na_exp_fp)
            samples_fp.sort(key=lambda x: int(x.split("_")[1]))

            # experiment stats
            num_str_correct = 0
            num_accept_correct = 0

            with open(results_fp, 'a+') as r:
                for sample_num, sample_fp in enumerate(samples_fp):
                    full_sample_fp = os.path.join(na_exp_fp, sample_fp)
                    label_line = labels.readline()
                    # if sample_num != 71:
                    #     continue
                    true_string, accept = label_line.split(",")
                    accept = bool(int(accept)) # convert from str -> bool
                    na_predicted_str, na_accept = neurosymbolic_automaton(full_sample_fp, regex)
                    print(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {na_predicted_str}={"accept" if na_accept else "reject"}')
                    r.write(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {na_predicted_str}={"accept" if na_accept else "reject"}\n')

                    if true_string == na_predicted_str:
                        num_str_correct += 1
                    
                    if accept == na_accept:
                        num_accept_correct += 1

                print(f'String Classification Accuracy: {(num_str_correct / 10) * 100}%, Task Accuracy: {(num_accept_correct / 10) * 100}%\n\n')
                r.write(f'String Classification Accuracy: {(num_str_correct / 10) * 100}%, Task Accuracy: {(num_accept_correct / 10) * 100}%')

                # add to totals
                total_str_correct += num_str_correct
                total_accept_correct += num_accept_correct

    with open(os.path.join(os.getcwd(), 'examples', 'regex', 'results', 'na', 'totals.txt'), 'a+') as r:
        print(f'String Classification Accuracy: {total_str_correct}%, Task Accuracy: {total_accept_correct}%\n\n')
        r.write(f'String Classification Accuracy: {total_str_correct}%, Task Accuracy: {total_accept_correct}%')

def get_vlm_output():
    """Run all experiments with the VLMs."""
    stitched_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', 'vlm', 'stitched')
    # models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']
    # models = ['llava-llama3']
    # models = ['llava:7b']
    models = ['moondream']

    for experiment_name in os.listdir(stitched_fp):
        if '5' in experiment_name or '7' in experiment_name or '8' in experiment_name or '9' in experiment_name or '2' in experiment_name or '6' in experiment_name or '0' in experiment_name:
            continue

        print(f'working on {experiment_name.split("__")[0]}')
        labels_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', f'{experiment_name}.txt')
        stitched_exp_fp = os.path.join(stitched_fp, experiment_name)

        # get the regex from the experiment name
        regex = experiment_name.split("__")[1]

        with open(labels_fp, 'r') as labels:
            # sort the samples
            samples_fp = os.listdir(stitched_exp_fp)
            samples_fp.sort(key=lambda x: int(x.split("_")[1]))

            for model in models:
                if not os.path.isdir(os.path.join('examples', 'regex', 'results', 'vlm', 'stitched', model)):
                    os.mkdir(os.path.join('examples', 'regex', 'results', 'vlm', 'stitched', model))
                results_fp = os.path.join('examples', 'regex', 'results', 'vlm', 'stitched', model, f'{experiment_name}.txt')
                with open(results_fp, 'a+') as r:
                    for sample_num, sample_fp in enumerate(samples_fp):
                        if (sample_num == 5 and '1' in experiment_name) or (sample_num == 6 and '1' in experiment_name) or (sample_num == 7 and '1' in experiment_name) or (sample_num == 8 and '1' in experiment_name):
                            print(f'#{sample_num} -> timeout')
                            r.write(f'#{sample_num} -> timeout\n')
                            label_line = labels.readline()
                            continue
                        full_sample_fp = os.path.join(stitched_exp_fp, sample_fp)
                        label_line = labels.readline()
                        if label_line == '':
                            continue
                        true_string, accept = label_line.split(",")
                        accept = bool(int(accept)) # convert from str -> bool
                        res1, time1, res2, time2 = vlm(model, full_sample_fp, regex)
                        print(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1} | {res2} : {time2}')
                        r.write(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1} | {res2} : {time2}\n')


def get_vlm_output_sequence():
    """Run all experiments with the VLMs given a sequence of images."""
    sequence_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', 'vlm', 'sequence')
    # models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']
    # models = ['llava-llama3']
    # models = ['llava:7b']
    # models = ['moondream']
    models = ['bakllava']

    for experiment_name in os.listdir(sequence_fp):
        # for skipping broken samples
        # if '8' in experiment_name or \
            # '5' in experiment_name or \
            # '7' in experiment_name or \
            # '9' in experiment_name or \
            # '2' in experiment_name or \
            # '0' in experiment_name or \
            # '6' in experiment_name: # or \
            # '1' in experiment_name:  #  or \
        #     '3' in experiment_name: 
            # continue

        print(f'working on {experiment_name.split("__")[0]}')
        labels_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', f'{experiment_name}.txt')
        sequence_exp_fp = os.path.join(sequence_fp, experiment_name)

        # get the regex from the experiment name
        regex = experiment_name.split("__")[1]

        with open(labels_fp, 'r') as labels:
            # sort the samples
            samples_fp = os.listdir(sequence_exp_fp)
            samples_fp.sort(key=lambda x: int(x.split("_")[1]))

            for model in models:
                if not os.path.isdir(os.path.join('examples', 'regex', 'results', 'vlm', 'sequence', model)):
                    os.mkdir(os.path.join('examples', 'regex', 'results', 'vlm', 'sequence', model))
                results_fp = os.path.join('examples', 'regex', 'results', 'vlm', 'sequence', model, f'{experiment_name}.txt')
                with open(results_fp, 'a+') as r:
                    for sample_num, sample_fp in enumerate(samples_fp):
                        # for skipping broken samples
                        # if (sample_num == 2 and '1' in experiment_name):
                        # if (sample_num in [2, 3, 6] and '1' in experiment_name):
                        #     print(f'#{sample_num} -> timeout')
                        #     r.write(f'#{sample_num} -> timeout\n')
                        #     label_line = labels.readline()
                        #     continue
                        full_sample_fp = os.path.join(sequence_exp_fp, sample_fp)

                        # full_sample_fp is now a directory for sequence
                        full_sample_fps = os.listdir(full_sample_fp)
                        full_sample_fps = [os.path.join(full_sample_fp, sfp) for sfp in full_sample_fps]
                        full_sample_fps.sort()

                        label_line = labels.readline()
                        if label_line == '':
                            continue
                        true_string, accept = label_line.split(",")
                        accept = bool(int(accept)) # convert from str -> bool
                        res1, time1 = vlm(model, full_sample_fps, regex)
                        # print(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1} | {res2} : {time2}')
                        # r.write(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1} | {res2} : {time2}\n')
                        print(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1}')
                        r.write(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1}\n')


def get_vlm_output2():
    """Run all experiments with the VLMs."""
    stitched_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', 'vlm', 'stitched2')
    models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']

    for experiment_name in os.listdir(stitched_fp):
        print(f'working on {experiment_name.split("__")[0]}')
        labels_fp = os.path.join(os.getcwd(), 'examples', 'regex', 'data', f'{experiment_name}.txt')
        stitched_exp_fp = os.path.join(stitched_fp, experiment_name)

        # get the regex from the experiment name
        regex = experiment_name.split("__")[1]

        with open(labels_fp, 'r') as labels:
            # sort the samples
            samples_fp = os.listdir(stitched_exp_fp)
            samples_fp.sort(key=lambda x: int(x.split("_")[1]))

            for model in models:
                if not os.path.isdir(os.path.join('examples', 'regex', 'results', 'vlm', 'stitched', model)):
                    os.mkdir(os.path.join('examples', 'regex', 'results', 'vlm', 'stitched2', model))
                results_fp = os.path.join('examples', 'regex', 'results', 'vlm', 'stitched2', model, f'{experiment_name}.txt')
                with open(results_fp, 'a+') as r:
                    for sample_num, sample_fp in enumerate(samples_fp):
                        full_sample_fp = os.path.join(stitched_exp_fp, sample_fp)
                        label_line = labels.readline()
                        if label_line == '':
                            continue
                        true_string, accept = label_line.split(",")
                        accept = bool(int(accept)) # convert from str -> bool

                        try:
                            res1, time1, res2, time2 = vlm(model, full_sample_fp, regex)
                        except TimeoutError:
                            res1, time1 = "timeout", TIMEOUT
                        finally:
                            signal.alarm(0)
                        
                        
                        
                        print(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1} | {res2} : {time2}')
                        r.write(f'#{sample_num} -> {true_string}={"accept" if accept else "reject"} | {res1} : {time1} | {res2} : {time2}\n')







if __name__=="__main__":
    # get_na_output()

    # get_vlm_output()

    get_vlm_output_sequence()