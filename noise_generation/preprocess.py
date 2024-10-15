import json

def add_noise_to_json(main_data, noise_data):
    # Create a dictionary for quick lookup of noise based on paragraphs
    noise_lookup = {item['paragraph']: item['noise'] for item in noise_data if 'noise' in item}
    
    # Loop through the main data and add the noise from noise_lookup
    for instance in main_data:
        paragraph = instance.get('paragraph', None)
        # If paragraph matches, add the noise key
        if paragraph in noise_lookup:
            instance['noise'] = noise_lookup[paragraph]
    
    return main_data

# Example usage
def merge_noise_into_json(main_json_path, noise_json_path, output_json_path):
    # Load the main JSON file
    with open(main_json_path, 'r') as main_file:
        main_data = json.load(main_file)
    
    # Load the noise JSON file
    with open(noise_json_path, 'r') as noise_file:
        noise_data = json.load(noise_file)
    
    # Add noise from the second file into the main file
    updated_data = add_noise_to_json(main_data, noise_data)
    
    # Write the updated data to a new JSON file
    with open(output_json_path, 'w') as output_file:
        json.dump(updated_data, output_file, indent=4)

# File paths
main_json = '/workspace/saradhi_pg/Nil/Work_3/datasets/updated_aaec_para_v2/updated_aaec_para_v2_test.json'
noise_json = '/workspace/saradhi_pg/Nil/Work_3/datasets/updated_aaec_para/updated_aaec_para_noise_test.json'
output_json = '/workspace/saradhi_pg/Nil/Work_3/datasets/updated_aaec_para_v2/updated_aaec_para_v2_noise_test.json'

# Merge noise and save to output
merge_noise_into_json(main_json, noise_json, output_json)
