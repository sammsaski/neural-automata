import os
import re


def analyze_na_results():
    results_fp = '/examples/regex/results/na'

    for exp_fp in os.listdir(results_fp):
        with open(os.path.join(results_fp, exp_fp), 'r') as f:
            correct = 0
            total = 0
            total_time_taken = 0

            for line in f:
                elements = line.split(" ")
                true_expression, true_value = elements[2].split("=")
                predicted_value = elements[4][:-1]
                time_taken = elements[5]

                total_time_taken += float(time_taken)

                if float(true_value) == float(predicted_value):
                    correct += 1
                total += 1
            
            print(f'{operand_length} Accuracy: {round((correct / total) * 100, 1)}%, Avg. Running Time: {total_time_taken / total}')


def parse_vlm_results(filepath):
    """
    Parse the output results files for the VLMs.
    """
    correct = 0
    total = 0
    running_time = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.split(r'#(\d+) -> ', content)
    
    parsed_data = {}
    for i in range(1, len(entries), 2):
        key = int(entries[i])
        value = entries[i+1].strip()

        # parse the value some more to break it up into: true expression,
        # true value, VLM output
        if value == "timeout":
            running_time += 300.0
            total += 1
            continue

        true_sample, vlm_output = value.split(" | ")
        true_expression, true_value = true_sample.split("=")
        vlm_runtime = vlm_output.split(": ")[-1]
        running_time += float(vlm_runtime)
        
        # if we detect the true value anywhere in the VLM output
        # then mark as correct for now
        if true_value == vlm_output:
            # print(true_value)
            # print(vlm_output)
            correct += 1
        total += 1

        parsed_data[key] = [true_expression, true_value, vlm_output]
    
    return parsed_data, correct / total, running_time / total


def analyze_vlm_results():
    results_fp = 'examples/regex/results/vlm/sequence'
    models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']

    for model in models:
        model_fp = os.path.join(results_fp, model)
        experiments = os.listdir(model_fp)
        experiments.sort()
        for i, exp_fp in enumerate(experiments):
            with open(os.path.join(model_fp, exp_fp), 'r') as f:
                parsed_items, accuracy, avg_runtime = parse_vlm_results(os.path.join(model_fp, exp_fp))
                print(f'Experiment #{i} | Task Accuracy: {accuracy}%, Avg. Running Time: {avg_runtime}')


# def analyze_vlm_results():
#     results_fp = 'examples/regex/results/vlm/sequence'
#     models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']

#     for model in models:
#         model_fp = os.path.join(results_fp, model)
#         experiments = os.listdir(model_fp)
#         experiments.sort()
#         for i, exp_fp in enumerate(experiments):
#             with open(os.path.join(model_fp, exp_fp), 'r') as f:
#                 correct = 0
#                 total = 0
#                 total_time_taken = 0

#                 for line in f:
#                     line = line[6:]
#                     if line == 'timeout\n':
#                         total_time_taken += 300.0
#                         total += 1
#                         continue
#                     elements = line.split(" | ")
#                     # print(f'wow: {line}\n\n\n')
#                     # print(f'hello: {elements[0]}\n\n\n')
#                     true_expression, true_value = elements[0].split("=")
#                     predicted_value, time_taken = elements[1].split(" : ")

#                     total_time_taken += float(time_taken)

#                     if true_value == predicted_value:
#                         correct += 1
#                     total += 1
                
#                 print(f'Experiment #{i} | Task Accuracy: {round((correct / total) * 100, 1)}%, Avg. Running Time: {total_time_taken / total}')

            
analyze_vlm_results()