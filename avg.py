import os
import json
import numpy as np
import glob
import re

path = "/workspace/saradhi_pg/Nil/Work_3/results/updated_cdcp-Key-Phrases-LAST-google/flan-t5-base-steps20000-SL1024-BS8"

directory_pattern = f'{path}/run_*/best_checkpoint/'

# Get all matching directories
directories = glob.glob(directory_pattern)

# Function to calculate mean and std or return NaN if the list is empty
def calc_mean_std(values):
    if len(values) > 0:
        return [float(np.mean(values)), float(np.std(values))]
    else:
        return [float('NaN'), float('NaN')]

# Function to process files based on a filename pattern
def calculate_scores(file_pattern):
    # Initialize lists to store the values
    component_precision_list = []
    component_recall_list = []
    component_f1_list = []
    relation_precision_list = []
    relation_recall_list = []
    relation_f1_list = []
    component_macro_f1_list = []

    # Iterate through the directories and their JSON files, and extract the values
    for directory in directories:
        for filename in os.listdir(directory):
            if file_pattern.match(filename):  # Only process files matching the pattern
                with open(os.path.join(directory, filename)) as f:
                    data = json.load(f)
                    component_precision_list.append(data.get("component_precision", float('NaN')))
                    component_recall_list.append(data.get("component_recall", float('NaN')))
                    component_f1_list.append(data.get("component_f1", float('NaN')))
                    relation_precision_list.append(data.get("relation_precision", float('NaN')))
                    relation_recall_list.append(data.get("relation_recall", float('NaN')))
                    relation_f1_list.append(data.get("relation_f1", float('NaN')))
                    component_macro_f1_list.append(data.get("component_macro_f1", float('NaN')))

    # Calculate the mean and standard deviation for each metric
    component_precision_stats = calc_mean_std(component_precision_list)
    component_recall_stats = calc_mean_std(component_recall_list)
    component_f1_stats = calc_mean_std(component_f1_list)
    relation_precision_stats = calc_mean_std(relation_precision_list)
    relation_recall_stats = calc_mean_std(relation_recall_list)
    relation_f1_stats = calc_mean_std(relation_f1_list)
    component_macro_f1_stats = calc_mean_std(component_macro_f1_list)

    # Create the output JSON structure
    output_data = {
        "num_runs": len(component_precision_list),
        "component_precision": component_precision_stats,
        "component_recall": component_recall_stats,
        "component_f1": component_f1_stats,
        "relation_precision": relation_precision_stats,
        "relation_recall": relation_recall_stats,
        "relation_f1": relation_f1_stats,
        "component_macro_f1": component_macro_f1_stats
    }

    return output_data

# Define patterns for different types of files
score_number_pattern = re.compile(r'^score_\d+\.json$')
# score_with_noise_pattern = re.compile(r'^score_with_noise\.json$')

# Calculate scores for different patterns
average_scores_number = calculate_scores(score_number_pattern)
# average_scores_number = calculate_scores(score_with_noise_pattern)


# Save the output JSON files
output_dir = f'{path}'
with open(os.path.join(output_dir, 'average_scores.json'), 'w') as f:
    json.dump(average_scores_number, f, indent=4)

print(f"Average scores with standard deviation saved for all patterns in {output_dir}")
