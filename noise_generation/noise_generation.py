import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name = "llama-3.1-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def generate_noise(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(inputs.input_ids, max_length=2048, num_return_sequences=1, repetition_penalty=1.1, temperature=0.5)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return explanation

def construct_new_instance(para):
    paragraph = para['paragraph']
    new_instance = f'Paragraph:\n{paragraph}'
    return new_instance


# Define the template for the prompt
prompt_template = """
Given an argumentative paragraph, generate a noisy sentence in 5-20 words only, which contradicts this paragraph. Here are the following examples for your reference.

Paragraph:
Technology has brought significant advantages to education. With the help of modern gadgets and the internet, students now have access to unlimited resources that can enhance their learning experience. Online courses and educational apps provide flexibility, allowing students to study at their own pace and revisit difficult topics. Technology also encourages collaboration through platforms like Google Classroom, where students can work together on assignments remotely. Furthermore, interactive learning tools such as simulations and educational games make complex subjects more understandable and engaging. As technology continues to advance, its role in education will only grow, making learning more accessible and effective.

Noisy Sentence:
Technology has no positive impact on education or learning.

Paragraph:
Exercise is essential for maintaining good health. Regular physical activity helps to reduce the risk of many chronic diseases, including heart disease, diabetes, and obesity. It also improves mental health by reducing symptoms of depression and anxiety while boosting overall mood. Moreover, exercise promotes better sleep and increases energy levels, making daily tasks easier to manage. Whether it's running, swimming, or practicing yoga, staying active is a key factor in living a long and healthy life. Everyone should incorporate at least 30 minutes of exercise into their daily routine for optimal well-being

Noisy Sentence:
Exercise is unnecessary and offers no health benefits at all.

Paragraph:
Sustainable energy is the key to a better future. With climate change threatening the planet, we must transition from fossil fuels to renewable energy sources like solar, wind, and hydropower. These alternatives produce far fewer greenhouse gases, reducing our carbon footprint and helping to combat global warming. Additionally, sustainable energy is cost-effective in the long run, as renewable resources are abundant and will not run out. By investing in renewable energy infrastructure today, we can create a cleaner, healthier environment for future generations while also fostering economic growth in green industries.

Noisy Sentence:
Sustainable energy has no effect on the environment and isn't worth pursuing.

Paragraph:
Reading books is one of the best ways to develop the mind and enhance knowledge. Books offer a wealth of information, provide deep insights, and stimulate critical thinking. Unlike short online articles or videos, books allow for a comprehensive exploration of subjects, fostering a greater understanding of various topics. Furthermore, reading improves focus, concentration, and vocabulary, all of which contribute to better communication skills. Whether fiction or non-fiction, books open up new worlds and perspectives, making them indispensable tools for intellectual growth and personal development.

Noisy Sentence:
Reading books wastes time and doesn’t improve knowledge or skills.

Paragraph:
Healthy eating is crucial for maintaining a balanced lifestyle and preventing disease. A diet rich in fruits, vegetables, whole grains, and lean proteins provides the body with essential nutrients that support overall well-being. Proper nutrition strengthens the immune system, boosts energy levels, and reduces the risk of chronic conditions such as heart disease, diabetes, and obesity. Moreover, healthy eating improves mental clarity and mood, promoting both physical and psychological health. By making mindful food choices, we can ensure a healthier, longer life.

Noisy Sentence:
Healthy eating has no effect on health and is unnecessary for well-being.

#### Real Example ####

{}

Noisy Sentence:

Note: Give the sentence only without any prefix. No code please.

"""

dataset_names = ['cdcp']
splits = ['train', 'dev', 'test']

for dataset in dataset_names:
    for split in splits:

        input_filename = f"./data/updated_{dataset}/updated_{dataset}_{split}.json"
        output_filename = f"./data/updated_{dataset}/updated_{dataset}_noise_{split}.json"

        with open(input_filename, 'r') as f:
            data = json.load(f)

        for idx, item in enumerate(tqdm(data, desc="Processing")):
            new_instance = construct_new_instance(item)
            prompt = prompt_template.format(new_instance)

            generated_text = generate_noise(prompt)
            # print (generated_text)
            noise = generated_text.split("Note: Give the sentence only without any prefix. No code please.")[-1].strip().split("\n")[0]
            print (noise)
            print ("---------------------------------------------------------------------------\n")
            item['noise'] = noise

        with open(output_filename, 'w') as output_file:
                json.dump(data, output_file, indent=4)
            
        print(f"Processed data saved to '{output_filename}'")
