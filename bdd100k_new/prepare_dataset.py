import json
import os
import shutil
from tqdm import tqdm
from PIL import Image
import argparse

# Paths to the dataset
train_annotation_file = '/home/alberto/datasets/bdd100k/labels/det_20/det_train.json'
val_annotation_file = '/home/alberto/datasets/bdd100k/labels/det_20/det_val.json'
train_image_directory = '/home/alberto/datasets/bdd100k/images/100k/train/'
val_image_directory = '/home/alberto/datasets/bdd100k/images/100k/val/'
output_directory = '/home/alberto/datasets/bdd100k/custom_dataset/'
test_image_directory = '/home/alberto/datasets/bdd100k/images/100k/test/'

# Desired combination
desired_combination = {
    'pedestrian': 5000,  # Minimum area in pixels for 'pedestrian'
    'bicycle': 5000,  # Minimum area in pixels for 'car',
    'rider': 5000
}
desired_size = 600


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/home/alberto/datasets/bdd100k/')
    parser.add_argument("--output-path", type=str, default='/home/alberto/datasets/bdd100k/custom_dataset/')
    args = parser.parse_args()
    return args


def prepare_custom_dataset(annotation_path, img_dir, output_dir, size):
    # Create output directories
    class_0_dir = os.path.join(output_dir, 'class_0')
    class_1_dir = os.path.join(output_dir, 'class_1')
    os.makedirs(class_0_dir, exist_ok=True)
    os.makedirs(class_1_dir, exist_ok=True)

    # Load annotations
    with open(annotation_path, 'r') as file:
        data = json.load(file)

    pbar = tqdm(total=len(data))
    # Process each image
    for item in data:
        image_name = item['name']
        if 'labels' not in item:
            pbar.update(1)
            continue
        classes_in_image = set()
        valid_for_class_1 = False

        for annotation in item['labels']:
            if 'category' in annotation and 'box2d' in annotation:
                class_name = annotation['category']
                bbox = annotation['box2d']
                width = abs(bbox['x2'] - bbox['x1'])
                height = abs(bbox['y2'] - bbox['y1'])
                area = width * height

                # Check area against threshold if the class is part of the desired combination
                if class_name in desired_combination and area >= desired_combination[class_name]:
                    valid_for_class_1 = True
                classes_in_image.add(class_name)

        # Check for desired combination
        if any([cls in classes_in_image for cls in desired_combination]) and valid_for_class_1:
            # Copy to class 1
            class_dir = class_1_dir
        else:
            # Copy to class 0
            class_dir = class_0_dir
        
        # Construct source and destination paths
        src_path = os.path.join(img_dir, image_name)
        dst_path = os.path.join(class_dir, image_name)

        with Image.open(src_path) as img:
            # Resize the image to be square
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img.save(dst_path)
        pbar.update(1)
    pbar.close()


def prepare_unlabeled_set(img_dir, output_dir, size):
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # List all images
    images = os.listdir(img_dir)

    pbar = tqdm(total=len(images))
    # Process each image
    for image_name in images:
        # Construct source and destination paths
        src_path = os.path.join(img_dir, image_name)
        dst_path = os.path.join(output_dir, image_name)

        with Image.open(src_path) as img:
            # Resize the image to be square
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img.save(dst_path)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    args = parse_args()
    # Run the dataset preparation
    # print('Preparing custom train dataset...')
    # prepare_custom_dataset(train_annotation_file, train_image_directory, f"{output_directory}/train", desired_size)
    # print('Preparing custom validation dataset...')
    # prepare_custom_dataset(val_annotation_file, val_image_directory, f"{output_directory}/val", desired_size)
    print('Preparing custom unlabeled dataset...')
    prepare_unlabeled_set(test_image_directory, f"{output_directory}/unlabeled", desired_size)