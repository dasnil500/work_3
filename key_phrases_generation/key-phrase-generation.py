import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

model_name = "/workspace/saradhi_pg/Nil/llama-experiment/llama-3.1-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def generate_entities(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(inputs.input_ids, max_length=2048, num_return_sequences=1, repetition_penalty=1.2, temperature = 0.1)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return explanation


def extract_first_json(content):
        
    brace_count = 0
    json_start = None
    json_end = None
    
    for i, char in enumerate(content):
        if char == '{':
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break
    
    if json_start is not None and json_end is not None:
        json_str = content[json_start:json_end]
        try:
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            pass
    
    return None

def jsonify_extracted_json(json_data):
    try:
        # Convert JSON object to JSON string with indentation
        json_string = json.dumps(json_data, indent=4)
        return json_string
    except (TypeError, ValueError) as e:
        return str(e)

# splits = ['train', 'dev', 'test']
splits = ['test']

for split in splits:

    input_filename = f"/workspace/saradhi_pg/Nil/llama-experiment/data/cdcp/cdcp_{split}.json"
    # output_filename = f"./data/aaec_para/aaec_para_key_phrases_{split}.json"

    with open(input_filename, 'r') as f:
        data = json.load(f)

    for idx, item in enumerate(tqdm(data, desc="Processing")):
        unique_id = item["id"]
        # paragraph = item['paragraph']
        components = item['components']
        relations = item['relations']

        for rel in relations:
            head_ac = components[rel['head']]['span']
            tail_ac = components[rel['tail']]['span']
            prompt = f"""
### Task: 
Given TWO related Arguments AC_1 and AC_2, generate key phrases of AC_1, generate key phrases of AC_2, generate related key phrases between AC_1 and AC_2 and the reason behind these related phrases step by step as follows:

### Steps:
- STEP 1: Identify key phrases of AC_1
- STEP 2: Identify key phrases of AC_2
- STEP 3: Identify list of related key phrases from AC_1 and AC_2.
- STEP 4: Give reason for each individual related key phrases.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Example 1:

Arguments:
  AC_1: "Affirmative action is good public policy."
  AC_2: "Diversity improves group decision-making."

Key Phrases of AC_1:
  - Affirmative action
  - Public policy

Key Phrases of AC_2:
  - Diversity
  - Decision-making

Related Key Phrases between AC_1 and AC_2: 
  - ("Public policy", "Decision-making")
  - ("Affirmative action", "Diversity")

Reason:
  - Decision-making enhances public policy.
  - Affirmative action increases diversity.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Example 2:

Arguments:
  AC_1: "Electric cars should be a priority to fight global warming."
  AC_2: "Public transportation is a better idea than electric cars."

Key Phrases of AC_1:
  - Electric cars
  - Global warming

Key Phrases of AC_2:
  - Public transportation
  - Electric cars

Related Key Phrases between AC_1 and AC_2 : 
  - ("Electric cars", "Global warming")
  - ("Electric cars", "Public transportation")

Reason:
  - To fight global warming, we must use electric cars.
  - Electric cars can be alternative to public transportation.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Example 3:

Arguments:
  AC_1: "What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others."
  AC_2: "Through cooperation, children can learn about interpersonal skills which are significant in the future life of all students."

Key Phrases of AC_1:
  - Team work
  - Achieve the same goal
  - Get along with others

Key Phrases of AC_2:
  - Cooperation
  - Interpersonal skills
  - Future life

Related Key Phrases between AC_1 and AC_2 : 
  - ("Team work", "Cooperation")
  - ("Get along with others", "Interpersonal skills")

Reason:
  - Teamwork improves cooperation.
  - Getting along with others requires interpersonal skills.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    

# Example 4:

Arguments:
  AC_1: "The significance of competition is that how to become more excellence to gain the victory."
  AC_2: "Competition makes the society more effective."

Key Phrases of AC_1:
  - Significance of competition
  - Become more excellence

Key Phrases of AC_2:
  - Competition
  - Society more effective

Related Key Phrases between AC_1 and AC_2 : 
  - ("Significance of competition", "Society more effective")
  - ("Become more excellence", "Competition")

Reason:
  - The significance of competition is that it makes the society more effective.
  - Competition drives individuals to become more excellent.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  

# Example 5:

Arguments:
  AC_1: "A healthy diet leads to a healthier body."
  AC_2: "A fit body has lots of energy and health benefits."

Key Phrases of AC_1:
  - Healthy diet
  - Healthier body

Key Phrases of AC_2:
  - Fit body
  - Energy
  - Health benefits

Related Key Phrases between AC_1 and AC_2 : 
  - ("Healthy diet", "Health benefits")
  - ("Healthy diet", "Fit body")
  - ("Healther body", "Energy")

Reason:
  - A healthy diet gives many health benefits.
  - Following a healthy diet helps you have a fitter body.
  - A healthier body has more energy. 

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  

### Real Data ###

Arguments:
  AC_1: "{head_ac}"
  AC_2: "{tail_ac}"
  
### Output:

#NOTE: Only give the output in the same format. No unnecessary tokens please.

"""            
            print (f'essay_id: {unique_id}\nAC_1: "{head_ac}"\nAC_2: "{tail_ac}\n"')
            generated_text = generate_entities(prompt)
            generated_text = generated_text.split("#NOTE: Only give the output in the same format. No unnecessary tokens please.")[-1].strip()
            generated_text = generated_text.split("\n\n\n")[0].strip()
            print (generated_text)
            print (f"--------------------------------------------------------------------------------------------------------\n")

            rel['key_phrases_text'] = generated_text
    
    # with open(output_filename, 'w') as f:
    #     json.dump(data, f, indent=4)
