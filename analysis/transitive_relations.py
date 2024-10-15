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
    
# ----------------------------------------------------------------------------------------------------------------


def detect_and_remove_transitive_relations(relations):
    # Build adjacency list
    adj_list = defaultdict(list)
    for rel in relations:
        adj_list[rel["head"]].append(rel["tail"])

    # Function to find all paths using DFS
    def find_paths(start, end, path):
        path = path + [start]
        if start == end:
            return [path]
        if start not in adj_list:
            return []
        paths = []
        for node in adj_list[start]:
            if node not in path:
                newpaths = find_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    # Find all paths
    all_paths = []
    for head in adj_list:
        for tail in adj_list[head]:
            all_paths.extend(find_paths(head, tail, []))

    # Identify and remove transitive relations
    transitive_relations = set()
    for path in all_paths:
        if len(path) > 2:
            for i in range(len(path) - 2):
                transitive_relations.add((path[i], path[i+2]))

    non_transitive_relations = []
    for rel in relations:
        if (rel["head"], rel["tail"]) not in transitive_relations:
            non_transitive_relations.append(rel)

    return transitive_relations, non_transitive_relations

# ------------------------------------------------------------------------------------------------------

def normalize_segment(segment):
    if segment is None:
        return ""
    return segment.strip().lower()

def find_chains_with_filter(relations):
    chains = []
    single_relations = []
    relation_dict = {}

    # for tail, relation_type, head in relations:
    #     tail = normalize_segment(tail)
    #     head = normalize_segment(head)
    #     if tail not in relation_dict:
    #         relation_dict[tail] = []
    #     relation_dict[tail].append((head, relation_type))

    for tail, head in relations:
        tail = normalize_segment(tail)
        head = normalize_segment(head)
        if tail not in relation_dict:
            relation_dict[tail] = []
        relation_dict[tail].append((head))
    
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
            for neighbor in relation_dict[current_node]:
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

# ----------------------------------------------------------------------------------

# datasets = ['cdcp', 'aaec_para', 'aaec_para_v2']
datasets = ['cdcp']
model_variants = ['base', 'xl', 'xxl']

for dataset in datasets:
    for variant in model_variants:
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

        pred_rel_set = set()

        for predicted_output in predicted_outputs:
            pred_matches = parse_output(predicted_output)
            if pred_matches:
                for i in pred_matches:
                    head_ac = i[0].lower().strip()
                    tail_ac = i[2].lower().strip()
                    pred_rel_set.add((head_ac, tail_ac))

        # print (pred_rel_set)
        # print ("#########################################################")

        json_filename = f"/workspace/saradhi_pg/Nil/Work_3/datasets/updated_{dataset}/updated_{dataset}_test.json"
        with open(json_filename, "r") as json_file:
            test_split = json.load(json_file)

        # gt_non_trans_rel = set()
        gt_trans_rel = set()
        filtered_chains = []
        single_relations = []

        for item in test_split:
            trans_rel, non_trans_rel = detect_and_remove_transitive_relations(item["relations"])

            # if (non_trans_rel):
            #     for i in non_trans_rel:
            #         head_index = i["head"]
            #         tail_index = i["tail"]
            #         rel_type = i["type"]
            #         rel_tuple = (head_index, tail_index)

            #         head_ac = item["components"][rel_tuple[0]]["span"].lower().strip()
            #         tail_ac = item["components"][rel_tuple[1]]["span"].lower().strip()
            #         gt_non_trans_rel.add((head_ac, tail_ac))

            if (trans_rel):
                for i in trans_rel:
                    head_ac = item["components"][i[0]]["span"].lower().strip()
                    tail_ac = item["components"][i[1]]["span"].lower().strip()
                    gt_trans_rel.add((head_ac,tail_ac))

            # filtered_chains, single_relations = find_chains_with_filter(gt_non_trans_rel.union(gt_trans_rel))

        correct_trans_rel = gt_trans_rel.intersection(pred_rel_set)
        if len(gt_trans_rel) > 0:
            percentage = len(correct_trans_rel) / len(gt_trans_rel) * 100
        else:
            percentage = 0

        print(f"\nNumber of GT transitive relations in {dataset} with {variant} variant: {len(gt_trans_rel)}")
        print(f"Number of Correct transitive relations in {dataset} with {variant} variant: {len(correct_trans_rel)}")
        print (f"Percentage of Correct Transitive Relations in {dataset} with {variant} variant: {percentage:.2f}%")





