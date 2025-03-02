import os
import random
import re
import string

import rstr
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets



def matches_regex(pattern, text):
  """
  Checks if the given text matches the regex pattern.

  Args:
    pattern: The regular expression pattern to match.
    text: The string to check against the pattern.

  Returns:
    True if the text matches the pattern, False otherwise.
  """
  match = re.fullmatch(pattern, text)
  return bool(match)


def generate_random_regex(length=10):
    """Generates a random regular expression of specified length."""
    
    regex = ""
    for _ in range(length):
        # Randomly choose a character type
        char_type = random.choices(
            population=["literal", "any", "range"],
            weights=[0.5, 0.25, 0.25]
        )[0] # have to unpack because output is list

        if char_type == "literal":
            regex += random.choice(string.ascii_lowercase)
        elif char_type == "any":
            regex += "[A-Za-z]"
        elif char_type == "range":
            c1 = random.choice(string.ascii_lowercase)
            c2 = random.choice(string.ascii_lowercase)
            start, end = sorted((c1, c2))
            regex += "[" + start + "-" + end + "]"
             
        # Add optional quantifier
        if random.random() < 0.2:
            regex += random.choice(["*", "+"])

    return regex


def generate_invalid_string(regex):
    """
    Randomly generate a string that will not match the given regex.
    """    
    # initialize s as a lowercase string with length 1-10
    s = rstr.rstr(string.ascii_lowercase, 1, 10)

    # keep randomly generating strings until we get an invalid one
    while matches_regex(regex, s):
        # allow for wider strings
        s = rstr.rstr(string.ascii_lowercase, 1, 30)

    return s


def generate_valid_string(regex, length_limit):
    """
    Randomly generate string that will match the given regex with
    limitations on the length of the string.
    """
    # initialize s
    s = rstr.xeger(rf'{regex}'.lower())

    while len(s) > length_limit:
        s = rstr.xeger(rf'{regex}'.lower())

    return s


def generate_strings(regex, num_strings):
    """
    Randomly generate 50 strings to be accepted and 50 to be rejected 
    by the given regular expression.
    """
    accept = [generate_valid_string(regex, length_limit=20) for _ in range(num_strings)]
    reject = [generate_invalid_string(regex) for _ in range(num_strings)]
    
    return accept, reject


def setup_experiment():
    """
    Generate the regular expressions and randomly generated strings for the experiments.
    Additionally, generate the images, tensors, and label files associated with all samples.
    """
    data_path = os.path.join('examples', 'regex', 'data')
    na_path = os.path.join(data_path, 'na')
    vlm_path = os.path.join(data_path, 'vlm')
    sequence_path = os.path.join(vlm_path, 'sequence')
    stitched_path = os.path.join(vlm_path, 'stitched')

    # load the testing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data = datasets.EMNIST(root="data/EMNIST", split="letters", train=True, download=True, transform=transform)
    print('finished downloading dataset! continuing...')
    images, labels = zip(*data)  # unzips dataset into separate tuples
    labels = [label-1 for label in labels] # set labels to be 0-labeled instead 1-labeled
    byclass = [[index for index, l in enumerate(labels) if l == target_label] for target_label in range(26)] # split dataset by class        

    # randomly choose the lengths of the regexs
    regex_lengths = [random.randint(3, 11) for _ in range(10)]

    # generate the regexs
    regexs = [generate_random_regex(length=length) for length in regex_lengths]
    print('finished generating regexs! continuing...')

    for i, regex in enumerate(regexs):
        # create directories for storing sample images
        regex_na_fp = os.path.join(na_path, f'experiment_{i}__{regex}')
        regex_sequence_fp = os.path.join(sequence_path, f'experiment_{i}__{regex}')
        regex_stitched_fp = os.path.join(stitched_path, f'experiment_{i}__{regex}') 
        labels_path = os.path.join(data_path, f'experiment_{i}__{regex}.txt')
        if not os.path.isdir(regex_na_fp):
            os.mkdir(regex_na_fp)
        if not os.path.isdir(regex_sequence_fp):
            os.mkdir(regex_sequence_fp)
        if not os.path.isdir(regex_stitched_fp):
            os.mkdir(regex_stitched_fp)
        
        # generate random strings
        accepted, rejected = generate_strings(regex, num_strings=50)
        sample_strings = accepted + rejected

        # write these accepted and rejected strings to a .txt file
        with open(labels_path, 'a+') as f:
            for accepted_string in accepted:
                f.write(f'{accepted_string}, 1\n')
            for rejected_string in rejected:
                f.write(f'{rejected_string}, 0\n')

        # create the images for them
        for j, sample_string in enumerate(sample_strings):
            string_na_path = os.path.join(regex_na_fp, f'sample_{j}__{sample_string}.pt') # tensor
            string_seq_path = os.path.join(regex_sequence_fp, f'sample_{j}__{sample_string}') # dir
            string_stitch_path = os.path.join(regex_stitched_fp, f'sample_{j}__{sample_string}.png') # image
            
            # create directories for sample string images (sequence only)
            if not os.path.isdir(string_seq_path):
                os.mkdir(string_seq_path)
                
            tensor_images = []
            modified_tensor_images = []
            for k, chr in enumerate(sample_string):
                # map the char to a label 0-26
                label = ord(chr) - ord('a')

                # get the image (tensor)
                image_index = random.choice(byclass[label])
                image = images[image_index]
                tensor_images.append(image)

                # create the png image and save it
                png_image = torch.fliplr(image)
                png_image = transforms.functional.rotate(png_image, angle=270)
                modified_tensor_images.append(png_image) # add after rotating and flipping
                png_image = transforms.ToPILImage()(png_image)
                png_image.save(os.path.join(string_seq_path, f'image_{k}__{chr}.png'))

            # create the stitched image and save it (for VLM)
            stitched_image = torch.hstack(modified_tensor_images)
            stitched_image = transforms.ToPILImage()(stitched_image)
            stitched_image.save(string_stitch_path)

            # save the tensor (for NSFA)
            sample_tensor = torch.stack(tensor_images)
            torch.save(sample_tensor, string_na_path)

        print(f'finished sample {i}!')


if __name__=="__main__":
    # regex = generate_random_regex()
    # print(regex)
    # a, r = generate_strings(regex, 1)
    # print(a)
    # print(r)

    data_path = os.path.join('examples', 'regex', 'data')
    na_path = os.path.join(data_path, 'na')
    vlm_path = os.path.join(data_path, 'vlm')
    sequence_path = os.path.join(vlm_path, 'sequence')
    stitched_path = os.path.join(vlm_path, 'stitched')

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    if not os.path.isdir(na_path):
        os.mkdir(na_path)

    if not os.path.isdir(vlm_path):
        os.mkdir(vlm_path)

    if not os.path.isdir(sequence_path):
        os.mkdir(sequence_path)

    if not os.path.isdir(stitched_path):
        os.mkdir(stitched_path)

    setup_experiment()
