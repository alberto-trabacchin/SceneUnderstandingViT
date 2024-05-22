import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Define combinations of interest here
combinations_of_interest = [
    ('car', 'traffic light')  # Example combination
]

# Load annotations and calculate various statistics
def calculate_statistics(annotation_file):
    with open(annotation_file, 'r') as file:
        data = json.load(file)
        object_counts = {}
        frame_counts = {}
        combination_counts = { '+'.join(combo): 0 for combo in combinations_of_interest }
        bbox_areas = {}

        for item in data:
            if 'labels' not in item:  # Continue if there are no labels
                continue
            classes_in_frame = set()
            for annotation in item['labels']:
                class_name = annotation['category']

                # Object and Frame Counts
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
                classes_in_frame.add(class_name)

                # Bounding Box Areas
                if 'box2d' in annotation:
                    bbox = annotation['box2d']
                    width = abs(bbox['x2'] - bbox['x1'])
                    height = abs(bbox['y2'] - bbox['y1'])
                    area = width * height
                    if class_name in bbox_areas:
                        bbox_areas[class_name].append(area)
                    else:
                        bbox_areas[class_name] = [area]

            # Update frame counts
            for class_name in classes_in_frame:
                if class_name in frame_counts:
                    frame_counts[class_name] += 1
                else:
                    frame_counts[class_name] = 1

            # Check for combinations in the frame
            for combo in combinations_of_interest:
                if all([c in classes_in_frame for c in combo]):
                    combination_counts['+'.join(combo)] += 1

        # Organize bbox area data for violin plot
        bbox_area_lists = [bbox_areas[key] for key in sorted(bbox_areas.keys())]
        medians = [np.median(areas) for areas in bbox_area_lists]
        
    return object_counts, frame_counts, bbox_area_lists, sorted(bbox_areas.keys()), medians, combination_counts

# Plotting function for all stats
def plot_statistics(object_counts, frame_counts, combination_counts, dataset_type):
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Plot object and frame counts with combination counts
    plt.figure(figsize=(18, 10))
    combined_counts = {**object_counts, **combination_counts}
    bars = plt.bar(combined_counts.keys(), combined_counts.values(), color='blue')
    plt.title(f'{dataset_type} Object and Combination Counts')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{dataset_type}_object_combination_counts.png'))
    plt.close()

# File paths
train_annotation_file = '/home/alberto-trabacchin-wj/datasets/bdd100k/labels/det_20/det_train.json'
val_annotation_file = '/home/alberto-trabacchin-wj/datasets/bdd100k/labels/det_20/det_val.json'

# Calculate and plot statistics for both train and validation datasets
train_object_counts, train_frame_counts, train_bbox_area_lists, train_class_names, train_medians, train_combination_counts = calculate_statistics(train_annotation_file)
val_object_counts, val_frame_counts, val_bbox_area_lists, val_class_names, val_medians, val_combination_counts = calculate_statistics(val_annotation_file)

plot_statistics(train_object_counts, train_frame_counts, train_combination_counts, 'Train')
plot_statistics(val_object_counts, val_frame_counts, val_combination_counts, 'Validation')
