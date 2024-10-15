import json
import re
import os

# File paths
json_file_path = "/workspace/saradhi_pg/Nil/llama-experiment/data/aaec_para_v2/aaec_para_v2_test.json"
text_file_path = "/workspace/saradhi_pg/Nil/llama-experiment/5.key_phrases_generation/key_phrases_test_aaec_para.txt"

# Load the JSON and text data
with open(json_file_path, "r") as f:
    json_data = json.load(f)

with open(text_file_path, "r") as f:
    text_data = f.read()

# Split the text data into instances
text_instances = text_data.split("--------------------------------------------------------------------------------------------------------")

def extract_related_key_phrases(related_phrases_text):
    # Find the start of the "Related Key Phrases" section
    start = related_phrases_text.find("Related Key Phrases between AC_1 and AC_2 :")
    if start == -1:
        return []

    # Extract the section containing related key phrases
    related_phrases_section = related_phrases_text[start:].split("Reason:")[0].strip()
    
    # Extract the key phrases from the section
    start = related_phrases_section.find(":")
    related_phrases = related_phrases_section[start+1:].strip().split("- ")
    
    # Clean and format each phrase pair into tuples
    related_key_phrases = []
    for phrase in related_phrases:
        if phrase:
            # Stop at the first closing parenthesis to remove any trailing text
            valid_phrase = re.search(r'\(.*?\)', phrase)
            if valid_phrase:
                phrase_tuple = tuple(map(lambda x: x.strip().strip('"'), valid_phrase.group(0).strip("()").split(", ")))
                if(len(phrase_tuple) == 2):
                    related_key_phrases.append(phrase_tuple)
    
    return related_key_phrases


# Process each instance and extract related key phrases
for item in text_instances:

    related_key_phrases = extract_related_key_phrases(item)

    lines = item.split("\n")

    AC_1 = lines[3][7:-1]
    AC_2 = lines[4][7:-1]

    print (f"AC1: {AC_1}\nAC2: {AC_2}\n")

    for instance in json_data:
        for relation in instance['relations']:
            head_component = instance['components'][relation['head']]['span']
            tail_component = instance['components'][relation['tail']]['span']

            # Match head and tail with AC_1 and AC_2
            if AC_1 == head_component and AC_2 == tail_component:
                relation['related_key_phrases'] = related_key_phrases
                print(f"Added Related Key-Phrases: {related_key_phrases}")
                break

print (json_data[1])

# Define the directory and file path
directory = "/workspace/saradhi_pg/Nil/llama-experiment/data/updated_aaec_para_v2"
file_path = os.path.join(directory, "updated_aaec_para_v2_test.json")

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Write the JSON data to the file
with open(file_path, "w") as f:
    json.dump(json_data, f, indent=4)
