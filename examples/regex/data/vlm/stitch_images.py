import sys
import os
import numpy as np
from PIL import Image

"""
For each sample directory coming frmo the 'sequence' examples, stitch the images for that sample together to make one image.
"""

def stitch_images(image_folder, output_image_path):
    # Get all .png files sorted by name (optional, but often useful)
    image_files = sorted([file for file in os.listdir(image_folder) if file.endswith('.png')])

    # Open all images
    images = [Image.open(os.path.join(image_folder, file)) for file in image_files]

    # Assume all images are the same size
    width, height = images[0].size

    # Create a new blank image with appropriate size
    total_width = width * len(images)
    new_image = Image.new('RGBA', (total_width, height))

    # Paste each image side by side
    for i, img in enumerate(images):
        new_image.paste(img, (i * width, 0))

    # Save the stitched image
    new_image.save(f'{output_image_path}.png')


if __name__=="__main__":
    cwd = os.getcwd() # neural-automata/examples/regex/data/vlm ; if run from vlm directory
    output_image_dir = os.path.join(cwd, 'stitched2')
    image_dir = os.path.join(cwd, 'sequence')

    for experiment_image_dir_name in os.listdir(image_dir):
        experiment_image_dir = os.path.join(image_dir, experiment_image_dir_name)

        os.makedirs(os.path.join(output_image_dir, experiment_image_dir_name))
        
        for sample_image_dir_name in os.listdir(experiment_image_dir):
            sample_image_dir = os.path.join(experiment_image_dir, sample_image_dir_name)
            output_image_path = os.path.join(output_image_dir, experiment_image_dir_name, sample_image_dir_name)

            stitch_images(sample_image_dir, output_image_path)

