import os
import re


def analyze_na_results():
    for operand_length in range(2, 11):
        with open(f'examples/simple_math_vlm_comp/results/na/{operand_length}/results.txt', 'r') as f:
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


def analyze_na_results2():
    for operand_length in range(2, 11):
        with open(f'examples/simple_math_vlm_comp/results/na2/{operand_length}/results.txt', 'r') as f:
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
        true_sample, vlm_output = value.split(" | ")
        true_expression, true_value = true_sample.split("=")
        # vlm_runtime = vlm_output.split(": ")[-1]
        vlm_runtime = vlm_output.split(": ")[-2]
        vlm_runtime = vlm_runtime.split(",")[0]
        running_time += float(vlm_runtime)
        
        # if we detect the true value anywhere in the VLM output
        # then mark as correct for now
        if true_value in vlm_output:
            # print(true_value)
            # print(vlm_output)
            correct += 1
        total += 1

        parsed_data[key] = [true_expression, true_value, vlm_output]
    
    return parsed_data, correct / total, running_time / total
         

def analyze_vlm_results():
    models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']
    for model in models:
        for operand_length in range(3, 11):
            fp = f'examples/simple_math_vlm_comp/results/sequence/{operand_length}/results_{model}.txt'
            _, accuracy, avg_runtime = parse_vlm_results(fp)
            print(f'{operand_length} {model} : {accuracy}, {avg_runtime}')

def analyze_vlm_results_stitched():
    models = ['llava-llama3', 'llava:7b', 'moondream', 'bakllava']
    for model in models:
        fp = f'examples/simple_math_vlm_comp/results/stitched_image/2/results_{model}.txt'
        _, accuracy, avg_runtime = parse_vlm_results(fp)
        print(f'{model} : {accuracy}, {avg_runtime}')


# for key, value in parsed_data.items():
#     print(f"Entry #{key}:\n{value}\n{'-'*40}")

# analyze_vlm_results()
# analyze_vlm_results_stitched()

# analyze_na_results()

analyze_na_results2()