import json
from collections import defaultdict

def parse_output(output):
    valid_matches = []
    depth = 0
    current_segment = ""
    inside_brackets = False
    
    for char in output:
        if char == '[':
            if depth > 0:
                # If already inside brackets, append this '[' to current_segment
                current_segment += char
            depth += 1
            inside_brackets = True
        elif char == ']':
            depth -= 1
            if depth == 0:
                # When depth returns to 0, we have a complete bracketed segment
                if inside_brackets:
                    parts = [part.strip() for part in current_segment.split('|')]
                    if len(parts) > 2:
                        for i in range(2, len(parts)):
                            result = (parts[0], parts[1], parts[i])
                            valid_matches.append(result)
                    current_segment = ""
                    inside_brackets = False
            else:
                # If still inside brackets, append this ']' to current_segment
                current_segment += char
        else:
            if inside_brackets:
                current_segment += char
    
    return valid_matches

def normalize_segment(segment):
    if segment is None:
        return ""
    return segment.strip().lower()

# Datasets and model variants
datasets = ['cdcp', 'aaec_para', 'aaec_para_v2']
model_variants = ['base', 'xl', 'xxl']

for dataset in datasets:
    for variant in model_variants:
        # Read the predicted outputs
        file_path = f'/workspace/saradhi_pg/Nil/Work_3/Analysis/inference_texts/{dataset}_flan-t5-{variant}_inference.txt'
        with open(file_path, 'r') as file:
            content = file.read()
        blocks = content.split("\n\n")
        true_outputs = []
        predicted_outputs = []
        for block in blocks:
            if block.startswith("True Output:"):
                true_outputs.append(block[len("True Output:"):].strip())
            elif block.startswith("Predicted Output:"):
                predicted_outputs.append(block[len("Predicted Output:"):].strip())

        # Load the test dataset
        json_filename = f"/workspace/saradhi_pg/Nil/Work_3/datasets/updated_{dataset}/updated_{dataset}_test.json"
        with open(json_filename, "r") as json_file:
            test_split = json.load(json_file)

        # Initialize dictionaries for counting
        distance_gt_counts = defaultdict(int)
        distance_correct_counts = defaultdict(int)
        distance_total_counts = defaultdict(int)

        # Ensure that predicted_outputs and test_split are aligned
        assert len(predicted_outputs) == len(test_split), "Mismatch in number of predicted outputs and test samples."

        # Process each item in the test dataset
        for idx, item in enumerate(test_split):
            predicted_output = predicted_outputs[idx]
            pred_matches = parse_output(predicted_output)

            # Build span_to_index mapping
            span_to_index = {}
            for comp_idx, comp in enumerate(item["components"]):
                normalized_span = comp["span"].lower().strip()
                span_to_index[normalized_span] = comp_idx

            # Build ground truth relations with distances
            gt_relations_by_distance = defaultdict(set)
            for rel in item["relations"]:
                head_idx = rel["head"]
                tail_idx = rel["tail"]
                distance = abs(head_idx - tail_idx) - 1
                head_ac = item["components"][head_idx]["span"].lower().strip()
                tail_ac = item["components"][tail_idx]["span"].lower().strip()
                gt_relations_by_distance[distance].add((head_ac, tail_ac))
                distance_gt_counts[distance] += 1

            # Process predicted relations
            for pred_rel in pred_matches:
                head_ac = pred_rel[0].lower().strip()
                tail_ac = pred_rel[2].lower().strip()

                # Map head and tail spans to indices
                if head_ac in span_to_index and tail_ac in span_to_index:
                    head_idx = span_to_index[head_ac]
                    tail_idx = span_to_index[tail_ac]
                    distance = abs(head_idx - tail_idx) - 1

                    # Update total predicted counts
                    distance_total_counts[distance] += 1

                    # Check if predicted relation is correct
                    if (head_ac, tail_ac) in gt_relations_by_distance[distance]:
                        distance_correct_counts[distance] += 1

        # Calculate and print accuracy per distance
        print(f"\nAnalysis for {dataset} dataset with {variant} variant:")
        for distance in sorted(distance_gt_counts.keys()):
            gt_count = distance_gt_counts[distance]
            correct_count = distance_correct_counts.get(distance, 0)
            total_pred_count = distance_total_counts.get(distance, 0)
            accuracy = (correct_count / gt_count) * 100 if gt_count > 0 else 0
            print(f"Distance {distance}: GT Relations = {gt_count}, Correctly Predicted = {correct_count}, Accuracy = {accuracy:.2f}%")
