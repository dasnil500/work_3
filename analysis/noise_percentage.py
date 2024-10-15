import json
import re

# Function to calculate noise percentage
def calculate_noise_percentage(instance):
    # Extract the paragraph and noise
    paragraph = instance["paragraph"]
    noise = instance.get("noise", "")
    
    # Split paragraph into sentences
    sentences = re.split(r'(?<=[.!?]) +', paragraph.strip())
    
    # Split noise into sentences
    noise_sentences = re.split(r'(?<=[.!?]) +', noise.strip())
    
    # Calculate noise percentage
    total_sentences = len(sentences)
    noise_sentence_count = len(noise_sentences)
    
    if total_sentences == 0:
        return 0, 0, 0
    
    noise_percentage = (noise_sentence_count / total_sentences) * 100
    return noise_percentage, total_sentences, noise_sentence_count

# Function to process the JSON file and accumulate the totals
def process_file(file_path):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        total_noise_percentage = 0
        total_sentences_count = 0
        total_noise_sentence_count = 0
        instance_count = len(json_data)

        for instance in json_data:
            noise_percentage, total_sentences, noise_sentence_count = calculate_noise_percentage(instance)
            total_noise_percentage += noise_percentage
            total_sentences_count += total_sentences
            total_noise_sentence_count += noise_sentence_count

        return total_noise_percentage, total_sentences_count, total_noise_sentence_count, instance_count

    except FileNotFoundError:
        print(f"File {file_path} not found. Please provide the correct path.")
        return 0, 0, 0, 0


datasets = ['aaec_para', 'aaec_para_v2', 'cdcp']

for dataset in datasets:
    # Paths to your train, test, and dev files
    train_file = f'/workspace/saradhi_pg/Nil/Work_3/datasets/updated_{dataset}/updated_{dataset}_noise_train.json'
    test_file = f'/workspace/saradhi_pg/Nil/Work_3/datasets/updated_{dataset}/updated_{dataset}_noise_test.json'
    dev_file = f'/workspace/saradhi_pg/Nil/Work_3/datasets/updated_{dataset}/updated_{dataset}_noise_dev.json'

    # Process each file and accumulate the results
    total_noise_percentage = 0
    total_sentences_count = 0
    total_noise_sentence_count = 0
    total_instance_count = 0

    for file_path in [train_file, test_file, dev_file]:
        noise_percentage, sentences_count, noise_sentence_count, instance_count = process_file(file_path)
        total_noise_percentage += noise_percentage
        total_sentences_count += sentences_count
        total_noise_sentence_count += noise_sentence_count
        total_instance_count += instance_count

    # Calculate the aggregated average
    if total_instance_count > 0:
        avg_noise_percentage = total_noise_percentage / total_instance_count
        avg_sentences = total_sentences_count / total_instance_count
        avg_noise_sentences = total_noise_sentence_count / total_instance_count

        # Compute the true aggregated noise percentage
        true_aggregated_noise_percentage = (total_noise_sentence_count / total_sentences_count) * 100

        print(f"\nAggregated Results ({dataset}):")
        print(f"Average noise percentage (individual instances): {avg_noise_percentage:.2f}%")
        print(f"Average total sentences: {avg_sentences:.2f}")
        print(f"Average noise sentences: {avg_noise_sentences:.2f}")
        print(f"True aggregated noise percentage: {true_aggregated_noise_percentage:.2f}%")
    else:
        print("No instances to process.")

