import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

max_seq_length = 512

# Load fine-tuned T5 model
model = T5ForConditionalGeneration.from_pretrained('./model')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Generate label-augmented text
input_text = "Smoking is injurious to health."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get model output
# with torch.no_grad():
#     # Generate label probabilities using T5's "text-to-text" format
#     logits = model.generate(input_ids)

output = model.generate(input_ids, max_length=max_seq_length, num_return_sequences=1, num_beams = 8)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated text:", generated_text)