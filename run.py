import json
import os
import shutil
import configparser
from utils import prepare_data, joint_decode_anl, relation_only_decode_anl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from evaluate import BatchEvaluator
import torch
from tqdm import tqdm
import torch.optim as optim
import argparse

# Read configurations from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Argument parsing setup
parser = argparse.ArgumentParser(description="Script for training and testing a model")

# Add arguments for each configuration value in config.ini
parser.add_argument('--do_train', type=bool, default=None, help="Whether to perform training")
parser.add_argument('--do_test', type=bool, default=None, help="Whether to perform testing")
parser.add_argument('--num_steps', type=int, default=None, help="Number of training steps")
parser.add_argument('--num_runs', type=int, default=None, help="Number of training runs")
parser.add_argument('--step_interval', type=int, default=None, help="Interval between checkpoint saves")
parser.add_argument('--learning_rate', type=float, default=None, help="Learning rate for optimizer")
parser.add_argument('--max_seq_length', type=int, default=None, help="Maximum sequence length")
parser.add_argument('--batch_size', type=int, default=None, help="Batch size for training")
parser.add_argument('--batch_size_inference', type=int, default=None, help="Batch size for inference")
parser.add_argument('--model_name_or_path', type=str, default=None, help="Pretrained model name or path")
parser.add_argument('--dataset_name', type=str, default=None, help="Name of the dataset to use")
parser.add_argument('--train_on_both', type=bool, default=None, help="Whether to train on both train and dev sets")
parser.add_argument('--variant_name', type=str, default=None, help="Variant name")
parser.add_argument('--resume_from_checkpoint', type=bool, default=None, help="Whether to resume from checkpoint")
parser.add_argument('--test_at_checkpoint', type=bool, default=None, help="Whether to test at each checkpoint")

args = parser.parse_args()

# Overwrite config values with command-line arguments, if provided
for key in vars(args):
    value = getattr(args, key)
    if value is not None:
        config.set('DEFAULT', key, str(value))

# Extract configuration values
do_train = config.getboolean('DEFAULT', 'do_train')
do_test = config.getboolean('DEFAULT', 'do_test')
num_steps = config.getint('DEFAULT', 'num_steps')
num_runs = config.getint('DEFAULT', 'num_runs')
step_interval = config.getint('DEFAULT', 'step_interval')
learning_rate = config.getfloat('DEFAULT', 'learning_rate')
max_seq_length = config.getint('DEFAULT', 'max_seq_length')
batch_size = config.getint('DEFAULT', 'batch_size')
batch_size_inference = config.getint('DEFAULT', 'batch_size_inference')
model_name_or_path = config.get('DEFAULT', 'model_name_or_path')
dataset_name = config.get('DEFAULT', 'dataset_name')
train_on_both = config.getboolean('DEFAULT', 'train_on_both')
variant_name = config.get('DEFAULT', 'variant_name')
resume_from_checkpoint = config.getboolean('DEFAULT', 'resume_from_checkpoint')
test_at_checkpoint = config.getboolean('DEFAULT', 'test_at_checkpoint')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels": self.target_encodings["input_ids"][idx],
        }

# Load data
def load_data(split):
    json_filename = f"./datasets/{dataset_name}/{dataset_name}_{split}.json"
    with open(json_filename, "r") as json_file:
        dataset = json.load(json_file)
    return dataset

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=max_seq_length)
def tokenize_data(input_sentences, target_sentences):
    input_encodings = tokenizer(input_sentences, padding=True, truncation=True, return_tensors="pt")
    target_encodings = tokenizer(target_sentences, padding=True, truncation=True, return_tensors="pt")
    return input_encodings, target_encodings

# Define perform_inference function outside do_train and do_test blocks
def perform_inference(model, dataloader, checkpoint_path=None):
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        evaluator = BatchEvaluator()
        for batch in tqdm(dataloader, desc="Evaluating on Test Data", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_test_loss += loss.item()

            # Generate predictions
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_seq_length)
            predicted_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            post_processed_outputs = [joint_decode_anl(pred) for pred in predicted_output]

            pred_components = [result[0] for result in post_processed_outputs]
            pred_relations = [result[1] for result in post_processed_outputs]

            # True Labels
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            true_outputs = [joint_decode_anl(label) for label in decoded_labels]

            true_components = [result[0] for result in true_outputs]
            true_relations = [result[1] for result in true_outputs]

            evaluator.add_batch(true_components, true_relations, pred_components, pred_relations)

        results = evaluator.evaluate()

        if checkpoint_path:
            file_name = os.path.join(checkpoint_path, "score.json")
            with open(file_name, 'w') as file:
                json.dump(results, file, indent=4)

        return results

def main():
    for run in range(1, num_runs + 1):
        print(f"Starting run {run}/{num_runs}")

        if do_train:
            train_dataset = load_data("train")
            val_dataset = load_data("dev")

            train_input_sentences, train_target_sentences = prepare_data(train_dataset)
            val_input_sentences, val_target_sentences = prepare_data(val_dataset)

            # Initialize model
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)

            if not train_on_both:
                train_input_encodings, train_target_encodings = tokenize_data(train_input_sentences, train_target_sentences)
                val_input_encodings, val_target_encodings = tokenize_data(val_input_sentences, val_target_sentences)

                train_dataset = CustomDataset(train_input_encodings, train_target_encodings)
                val_dataset = CustomDataset(val_input_encodings, val_target_encodings)

            if train_on_both:
                train_input_encodings, train_target_encodings = tokenize_data(train_input_sentences + val_input_sentences, train_target_sentences + val_target_sentences)
                train_dataset = CustomDataset(train_input_encodings, train_target_encodings)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

            # Create a linear learning rate scheduler
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: max(0, 1 - step / num_steps))

            def train(model, dataloader, optimizer, scheduler, step):
                model.train()
                total_loss = 0

                for batch in tqdm(dataloader, desc=f"Training Step {step}", leave=False):
                    optimizer.zero_grad()
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                return total_loss / len(dataloader)

            best_score = -1
            best_checkpoint = None

            # Function to find the latest checkpoint
            def find_latest_checkpoint(output_dir):
                checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                if not checkpoints:
                    return None
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
                return os.path.join(output_dir, checkpoints[-1])

            current_step = 0
            output_dir = f"./results/{dataset_name}-{variant_name}-{model_name_or_path}-steps{num_steps}-SL{max_seq_length}-BS{batch_size}/run_{run}"
            os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

            best_checkpoint_path = f"{output_dir}/best_checkpoint"
            if os.path.exists(best_checkpoint_path):
                # Load existing best score
                best_score_file = [f for f in os.listdir(best_checkpoint_path) if f.startswith("score_")]
                if best_score_file:
                    with open(os.path.join(best_checkpoint_path, best_score_file[0]), 'r') as file:
                        score_data = json.load(file)
                        best_score = score_data["component_macro_f1"] + score_data["relation_f1"]


            if resume_from_checkpoint:
                latest_checkpoint = find_latest_checkpoint(output_dir)
                if latest_checkpoint:
                    model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint).to(device)
                    optimizer.load_state_dict(torch.load(os.path.join(latest_checkpoint, "optimizer.pt")))
                    scheduler.load_state_dict(torch.load(os.path.join(latest_checkpoint, "scheduler.pt")))
                    current_step = int(latest_checkpoint.split("-")[-1])
                    print(f"Resuming from checkpoint: {latest_checkpoint} at step {current_step}")

            # Training and evaluating
            while current_step < num_steps:
                # Train
                avg_train_loss = train(model, train_dataloader, optimizer, scheduler, current_step)

                if test_at_checkpoint:
                    checkpoint_path = f"{output_dir}/checkpoint-{current_step + step_interval}"

                    os.makedirs(checkpoint_path, exist_ok=True)
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

                    print(f"Performing Inference on checkpoint: {checkpoint_path}")

                    results = perform_inference(model, val_dataloader if not train_on_both else train_dataloader, checkpoint_path)

                    combined_f1 = results["component_f1"] + results["relation_f1"]
                    if combined_f1 > best_score:
                        best_score = combined_f1
                        best_checkpoint = checkpoint_path

                        if os.path.exists(best_checkpoint_path):
                            shutil.rmtree(best_checkpoint_path)
                        model.save_pretrained(best_checkpoint_path)
                        tokenizer.save_pretrained(best_checkpoint_path)
                        torch.save(optimizer.state_dict(), os.path.join(best_checkpoint_path, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(best_checkpoint_path, "scheduler.pt"))

                        file_name = os.path.join(best_checkpoint_path, f"score_{current_step + step_interval}.json")
                        with open(file_name, 'w') as file:
                            json.dump(results, file, indent=4)

                        print(f"New best checkpoint saved at step {current_step + step_interval} with combined_f1: {combined_f1}")

                    # Clean up intermediate checkpoints
                    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                    for ckpt in checkpoints:
                        if ckpt != os.path.basename(best_checkpoint_path) and ckpt != os.path.basename(checkpoint_path):
                            shutil.rmtree(os.path.join(output_dir, ckpt))

                current_step += step_interval

            # Save latest checkpoint
            if current_step >= num_steps:
                latest_checkpoint_path = f"{output_dir}/checkpoint-{current_step}"
                model.save_pretrained(latest_checkpoint_path)
                tokenizer.save_pretrained(latest_checkpoint_path)
                torch.save(optimizer.state_dict(), os.path.join(latest_checkpoint_path, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(latest_checkpoint_path, "scheduler.pt"))
                print(f"Latest checkpoint saved at step {current_step}")

        if do_test:
            final_model_path = f"./results/{dataset_name}-{variant_name}-{model_name_or_path}-steps{num_steps}-SL{max_seq_length}-BS{batch_size}/run_{run}/best_checkpoint"
            print(f"Performing Inference on Best Model checkpoint: {final_model_path}")

            test_dataset = load_data("test")
            test_input_sentences, test_target_sentences = prepare_data(test_dataset)
            test_input_encodings, test_target_encodings = tokenize_data(test_input_sentences, test_target_sentences)

            test_dataset = CustomDataset(test_input_encodings, test_target_encodings)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size_inference, shuffle=False)

            model = AutoModelForSeq2SeqLM.from_pretrained(final_model_path).to(device)
            results = perform_inference(model, test_dataloader)

            file_name = os.path.join(final_model_path, "score.json")
            with open(file_name, 'w') as file:
                json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()
