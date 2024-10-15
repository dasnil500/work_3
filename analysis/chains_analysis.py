from collections import defaultdict
import os

def parse_output(output):
    valid_matches = []
    depth = 0
    current_segment = ""
    inside_brackets = False
    
    for char in output:
        if char == '[':
            if depth > 0:
                current_segment += char
            depth += 1
            inside_brackets = True
        elif char == ']':
            depth -= 1
            if depth == 0:
                if inside_brackets:
                    parts = [part.strip() for part in current_segment.split('|')]
                    # print (f"Output: {output}\n{len(parts)}:\n{parts}\n\n")
                    if len(parts) > 2:
                        for i in range(2, len(parts)):
                            result = (parts[0], parts[1], parts[i])
                            valid_matches.append(result)
                    current_segment = ""
                    inside_brackets = False
            else:
                current_segment += char
        else:
            if inside_brackets:
                current_segment += char
    
    return valid_matches


def normalize_segment(segment):
    if segment is None:
        return ""
    return segment.strip().lower()


def find_chains_with_filter(relations):
    chains = []
    single_relations = []
    relation_dict = {}

    for tail, relation_type, head in relations:
        tail = normalize_segment(tail)
        head = normalize_segment(head)
        if tail not in relation_dict:
            relation_dict[tail] = []
        relation_dict[tail].append((head, relation_type))
    
    for node in relation_dict:
        stack = [(node, [node])]
        while stack:
            current_node, current_chain = stack.pop()
            if current_node not in relation_dict:
                if len(current_chain) == 2:  # Track single relations (2 nodes, 1 relation)
                    single_relations.append(current_chain)
                elif len(current_chain) > 2:  # Track chains with more than 1 relation
                    chains.append(current_chain)
                continue
            for neighbor, etype in relation_dict[current_node]:
                if neighbor in current_chain:
                    continue
                stack.append((neighbor, current_chain + [neighbor]))
            if len(current_chain) > 2:  # Ensure only chains with >1 relation are kept
                chains.append(current_chain)
            elif len(current_chain) == 2:  # Keep track of single relations
                single_relations.append(current_chain)
    
    chains.sort(key=len, reverse=True)
    filtered_chains = []
    for chain in chains:
        is_subchain = any(
            set(chain).issubset(set(other_chain)) and chain != other_chain
            for other_chain in chains
        )
        if not is_subchain and len(chain) > 2:  # Only keep chains with >2 nodes
            filtered_chains.append(chain)

    return filtered_chains, single_relations


def calculate_common_chain_percentage(predicted_chains, true_chains_set):
    common_chains = predicted_chains.intersection(true_chains_set)
    common_percentage = (len(common_chains) / len(true_chains_set)) * 100 if true_chains_set else 0
    return common_percentage, common_chains


def extract_subchains(chain, min_length=2):
    subchains = []
    chain_length = len(chain)
    
    for length in range(min_length, chain_length + 1):
        for i in range(chain_length - length + 1):
            subchain = chain[i:i + length]
            subchains.append(tuple(subchain))
    
    return subchains

def analyze_chain_lengths(chains):
    length_distribution = defaultdict(int)
    
    for chain in chains:
        subchains = extract_subchains(chain)
        for subchain in subchains:
            length_distribution[len(subchain)] += 1
    
    return length_distribution


def calculate_lengthwise_accuracy(true_chains_set, predicted_chains_set):
    lengthwise_accuracy = {}
    
    true_subchains_by_length = defaultdict(set)
    predicted_subchains_by_length = defaultdict(set)

    for chain in true_chains_set:
        subchains = extract_subchains(chain)
        for subchain in subchains:
            true_subchains_by_length[len(subchain)].add(subchain)

    for chain in predicted_chains_set:
        subchains = extract_subchains(chain)
        for subchain in subchains:
            predicted_subchains_by_length[len(subchain)].add(subchain)

    for length in true_subchains_by_length:
        true_chains = true_subchains_by_length[length]
        predicted_chains = predicted_subchains_by_length.get(length, set())

        common_chains = true_chains.intersection(predicted_chains)

        correct_percentage = (len(common_chains) / len(true_chains)) * 100 if true_chains else 0
        lengthwise_accuracy[length] = correct_percentage

    return lengthwise_accuracy


datasets = ['cdcp', 'aaec_para', 'aaec_para_v2']
model_variants = ['base', 'xl', 'xxl']

for dataset in datasets:
    for variant in model_variants:
        print(f"\nProcessing Dataset: {dataset}, Model Variant: {variant} -----------------------------------------------")

        file_path = f'/workspace/saradhi_pg/Nil/Work_3/Analysis/inference_texts/{dataset}_flan-t5-{variant}_inference.txt'
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

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

        true_chains_set = set()
        predicted_chains_set = set()
        true_single_relations = set()
        predicted_single_relations = set()

        for true_output, predicted_output in zip(true_outputs, predicted_outputs):

            true_matches = parse_output(true_output)
            filtered_true_chains, true_singles = find_chains_with_filter(true_matches)
            # print (filtered_true_chains)
            
            for chain in filtered_true_chains:
                true_chains_set.add(tuple(chain))
            for single in true_singles:
                true_single_relations.add(tuple(single))
            
            pred_matches = parse_output(predicted_output)
            filtered_pred_chains, pred_singles = find_chains_with_filter(pred_matches)
            
            for chain in filtered_pred_chains:
                predicted_chains_set.add(tuple(chain))
            for single in pred_singles:
                predicted_single_relations.add(tuple(single))

        common_percentage, common_chains = calculate_common_chain_percentage(predicted_chains_set, true_chains_set)
        print(f"\nPercentage of EXACT MATCH Correct Chains of All Lengths: {common_percentage:.2f}%\n")

        true_length_distribution = analyze_chain_lengths(true_chains_set)
        print(f"True chains length distribution: {dict(true_length_distribution)}")

        predicted_length_distribution = analyze_chain_lengths(predicted_chains_set)
        print(f"Predicted chains length distribution: {dict(predicted_length_distribution)}")

        common_length_distribution = analyze_chain_lengths(common_chains)
        print(f"Correct chains length distribution: {dict(common_length_distribution)}\n")

        # print(f"Number of single relations in True Output: {len(true_single_relations)}")
        # print(f"Number of single relations in Predicted Output: {len(predicted_single_relations)}\n")

        lengthwise_accuracy = calculate_lengthwise_accuracy(true_chains_set, predicted_chains_set)
        for length, accuracy in sorted(lengthwise_accuracy.items()):
            print(f"Percentage of correctly predicted {length}-length chains: {accuracy:.2f}%")
