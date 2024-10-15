import re
import random

from collections import defaultdict

# def remove_transitive_relations(relations):
#     # Build adjacency list
#     adj_list = defaultdict(list)
#     for rel in relations:
#         adj_list[rel["head"]].append(rel["tail"])

#     # Function to find all paths using DFS
#     def find_paths(start, end, path):
#         path = path + [start]
#         if start == end:
#             return [path]
#         if start not in adj_list:
#             return []
#         paths = []
#         for node in adj_list[start]:
#             if node not in path:
#                 newpaths = find_paths(node, end, path)
#                 for newpath in newpaths:
#                     paths.append(newpath)
#         return paths

#     # Find all paths
#     all_paths = []
#     for head in adj_list:
#         for tail in adj_list[head]:
#             all_paths.extend(find_paths(head, tail, []))

#     # Identify and remove transitive relations
#     transitive_relations = set()
#     for path in all_paths:
#         if len(path) > 2:
#             for i in range(len(path) - 2):
#                 transitive_relations.add((path[i], path[i+2]))

#     non_transitive_relations = []
#     for rel in relations:
#         if (rel["head"], rel["tail"]) not in transitive_relations:
#             non_transitive_relations.append(rel)

#     return transitive_relations, non_transitive_relations



# def detect_and_add_transitive_relations(components, relations):
#     # Build adjacency list
#     adj_list = defaultdict(list)
#     span_to_index = {comp['span']: idx for idx, comp in enumerate(components)}

#     for rel in relations:
#         head_idx = span_to_index[rel[0][0]]
#         tail_idx = span_to_index[rel[2][0]]
#         adj_list[head_idx].append((tail_idx, rel[1]))

#     # print(adj_list)

#     # Function to find the longest path using DFS
#     def find_longest_paths():
#         def dfs(node, visited):
#             visited.append(node)
#             paths = []
#             for neighbor, rel_type in adj_list[node]:
#                 if neighbor not in visited:
#                     sub_paths = dfs(neighbor, visited.copy())
#                     for sub_path in sub_paths:
#                         paths.append([(node, neighbor, rel_type)] + sub_path)
#             if not paths:
#                 return [[]]
#             return paths
        
#         all_paths = []
#         # Convert keys to list to avoid runtime error during iteration
#         for start_node in list(adj_list.keys()):
#             all_paths.extend(dfs(start_node, []))

#         # Find the longest paths
#         max_length = max(len(path) for path in all_paths)
#         longest_paths = [path for path in all_paths if len(path) == max_length]

#         return longest_paths

#     longest_paths = find_longest_paths()

#     # print (longest_paths)

#     # Add transitive relations based on the longest paths
#     transitive_relations = set()
#     for path in longest_paths:
#         for i in range(len(path)):
#             for j in range(i+1, len(path)):
#                 head_idx = path[i][0]
#                 tail_idx = path[j][1]
#                 relation_type = path[i][2]  # Relation type from the first adjacent pair
#                 transitive_relations.add((head_idx, tail_idx, relation_type))

#     # print("\nTransitive Relations to be Added:\n")
#     # for head_idx, tail_idx, rel_type in transitive_relations:
#     #     print(f"({head_idx}, {tail_idx}, {rel_type})")

#     # Add these transitive relations to the original relations
#     for head_idx, tail_idx, relation_type in transitive_relations:
#         head_span = components[head_idx]['span']
#         head_type = components[head_idx]['type']
#         tail_span = components[tail_idx]['span']
#         tail_type = components[tail_idx]['type']
#         new_relation = ((head_span, head_type), relation_type, (tail_span, tail_type))
#         relations.append(new_relation)

#     return relations

def anl_input(text, components) -> str:

    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Generate the formatted input
    formatted_input = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_start, comp_end = comp['start'], comp['end']

        # Add text before the component span
        formatted_input += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_input += f"[ {component_text} "

        formatted_input += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_input += text[prev_end:]

    target_input = " ".join(formatted_input.split())  # Remove extra spaces

    # print (target_input)
    return target_input


def anl_input_with_noise(text, components, noise) -> str:
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    sorted_components = sorted(components, key=lambda x: x['start'])

    formatted_input = ""
    prev_end = 0  

    for comp in sorted_components:
        comp_start, comp_end = comp['start'], comp['end']

        formatted_input += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_input += f"[ {component_text} "

        formatted_input += "]"
        prev_end = comp_end

    formatted_input += text[prev_end:]

    target_input = " ".join(formatted_input.split())

    sentences = re.split(r'(?<=[\]])\s+', target_input)

    insert_position = random.randint(0, len(sentences) - 1)  # Randomly pick a position to insert the noise
    sentences.insert(insert_position + 1, noise.strip())  # Insert noise after a complete sentence

    noisy_input = " ".join(sentences)

    return noisy_input


# def anl_output(text, components, relations) -> str:

#     # _, relations = remove_transitive_relations(relations)

#     # Add IDs to components
#     index_counter = 0
#     for component in components:
#         component['id'] = index_counter
#         index_counter += 1

#     # Sort components by their start index
#     sorted_components = sorted(components, key=lambda x: x['start'])

#     # Create a dictionary to track which component has which relations
#     relation_dict = {}
#     for relation in relations:
#         relation_type = relation['type']
#         head = relation['head']
#         tail = relation['tail']
#         key_phrases = relation["related_key_phrases"]
#         if head not in relation_dict:
#             relation_dict[head] = []
#         relation_dict[head].append((relation_type, tail, key_phrases))

#     # Generate the formatted output
#     formatted_output = ""
#     prev_end = 0  # Track the end of the previous span

#     for comp in sorted_components:
#         comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

#         # Add text before the component span
#         formatted_output += text[prev_end:comp_start]

#         component_text = text[comp_start:comp_end]
#         formatted_output += f"[ {component_text} | {comp_type} "

#         # Add relations if any
#         if comp_index in relation_dict:
#             for relation_type, tail, key_phrase in relation_dict[comp_index]:
#                 tail_component = next(filter(lambda x: x['id'] == tail, components))
#                 tail_text = text[tail_component['start']:tail_component['end']]
#                 # formatted_output += f"| {relation_type.capitalize()} = {tail_text} "
#                 formatted_output += f"| {tail_text} "

#         formatted_output += "]"
#         prev_end = comp_end

#     # Add any remaining text after the last component
#     formatted_output += text[prev_end:]

#     target_output = " ".join(formatted_output.split())  # Remove extra spaces

#     print (target_output)
#     return target_output


def joint_anl_output(text, components, relations) -> str:

    # Add IDs to components
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Create a dictionary to track which component has which relations
    key_phrases_list = ""
    relation_dict = {}
    for relation in relations:
        relation_type = relation['type']
        head = relation['head']
        tail = relation['tail']
        key_phrases = relation.get("related_key_phrases", None)
        
        if key_phrases:
            first_phrase, second_phrase = key_phrases[0]
            key_phrases_list += f"({first_phrase}, {second_phrase}), "
        
        if head not in relation_dict:
            relation_dict[head] = []
        relation_dict[head].append((relation_type, tail))

    # Remove the trailing comma and space from the concatenated string, if it exists
    if key_phrases_list.endswith(", "):
        key_phrases_list = key_phrases_list[:-2]


    # Generate the formatted output
    formatted_output = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

        # Add text before the component span
        formatted_output += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_output += f"[ {component_text} | {comp_type} "

        # Add relations if any
        if comp_index in relation_dict:
            for relation_type, tail in relation_dict[comp_index]:
                tail_component = next(filter(lambda x: x['id'] == tail, components))
                tail_text = text[tail_component['start']:tail_component['end']]
                # formatted_output += f"| {relation_type.capitalize()} = {tail_text} "
                formatted_output += f"| {tail_text} "


        formatted_output += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_output += text[prev_end:]

    target_output = " ".join(formatted_output.split())  # Remove extra spaces

    if (key_phrases_list):
        return f'{target_output}\nStrongly Related Key Phrases: {key_phrases_list}'
    else:
        return target_output
    
def component_only_output(text, components, relations) -> str:

    # _, relations = remove_transitive_relations(relations)

    # Add IDs to components
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Create a dictionary to track which component has which relations
    key_phrases_list = ""
    relation_dict = {}
    for relation in relations:
        relation_type = relation['type']
        head = relation['head']
        tail = relation['tail']
        key_phrases = relation["related_key_phrases"]
        
        if key_phrases:
            first_phrase, second_phrase = key_phrases[0]
            key_phrases_list += f"({first_phrase}, {second_phrase}), "
        
        if head not in relation_dict:
            relation_dict[head] = []
        relation_dict[head].append((relation_type, tail))

    # Remove the trailing comma and space from the concatenated string, if it exists
    if key_phrases_list.endswith(", "):
        key_phrases_list = key_phrases_list[:-2]


    # Generate the formatted output
    formatted_output = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

        # Add text before the component span
        formatted_output += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_output += f"[ {component_text} | {comp_type} "
        formatted_output += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_output += text[prev_end:]

    target_output = " ".join(formatted_output.split())  # Remove extra spaces

    # if (key_phrases_list):
    #     return f'{target_output}\nStrongly Related Key Phrases: {key_phrases_list}'
    # else:
    #     return target_output

    return target_output
    

def relation_only_output(text, components, relations) -> str:

    # _, relations = remove_transitive_relations(relations)

    # Add IDs to components
    index_counter = 0
    for component in components:
        component['id'] = index_counter
        index_counter += 1

    # Sort components by their start index
    sorted_components = sorted(components, key=lambda x: x['start'])

    # Create a dictionary to track which component has which relations
    key_phrases_list = ""
    relation_dict = {}
    for relation in relations:
        relation_type = relation['type']
        head = relation['head']
        tail = relation['tail']
        key_phrases = relation["related_key_phrases"]
        
        if key_phrases:
            first_phrase, second_phrase = key_phrases[0]
            key_phrases_list += f"({first_phrase}, {second_phrase}), "
        
        if head not in relation_dict:
            relation_dict[head] = []
        relation_dict[head].append((relation_type, tail))

    # Remove the trailing comma and space from the concatenated string, if it exists
    if key_phrases_list.endswith(", "):
        key_phrases_list = key_phrases_list[:-2]


    # Generate the formatted output
    formatted_output = ""
    prev_end = 0  # Track the end of the previous span

    for comp in sorted_components:
        comp_index, comp_type, comp_start, comp_end = comp['id'], comp['type'], comp['start'], comp['end']

        # Add text before the component span
        formatted_output += text[prev_end:comp_start]

        component_text = text[comp_start:comp_end]
        formatted_output += f"[ {component_text} "

        # Add relations if any
        if comp_index in relation_dict:
            for relation_type, tail in relation_dict[comp_index]:
                tail_component = next(filter(lambda x: x['id'] == tail, components))
                tail_text = text[tail_component['start']:tail_component['end']]
                # formatted_output += f"| {relation_type.capitalize()} = {tail_text} "
                formatted_output += f"| {tail_text} "


        formatted_output += "]"
        prev_end = comp_end

    # Add any remaining text after the last component
    formatted_output += text[prev_end:]

    target_output = " ".join(formatted_output.split())  # Remove extra spaces

    if (key_phrases_list):
        return f'{target_output}\nStrongly Related Key Phrases: {key_phrases_list}'
    else:
        return target_output



def prepare_data(dataset):
    input_sentences = []
    target_sentences = []

    for example in dataset:
        input_sen = anl_input(example["paragraph"], example["components"])
        target_sen = joint_anl_output(example["paragraph"], example["components"], example["relations"])
        # print (f"Input:\n{input_sen}\n")
        # print (f"Target:\n{target_sen}\n")

        input_sentences.append(input_sen)
        target_sentences.append(target_sen)

    return input_sentences, target_sentences

def prepare_data_with_noise(dataset):
    input_sentences = []
    target_sentences = []

    for example in dataset:
        input_sen_with_noise = anl_input_with_noise(example["paragraph"], example["components"], example['noise'])
        target_sen = joint_anl_output(example["paragraph"], example["components"], example["relations"])
        print (f"Input:\n{input_sen_with_noise}\n")
        print (f"Output:\n{target_sen}\n")

        input_sentences.append(input_sen_with_noise)
        target_sentences.append(target_sen)

    return input_sentences, target_sentences



def joint_decode_anl(formatted_text):

    valid_comp_types = ['Premise', 'Claim', 'MajorClaim', 'value', 'fact', 'policy', 'reference', 'testimony', 'common_ground', 'hypothetical_instance', 'statistics', 'real_example', 'others']

    formatted_text = re.sub(r'\](\W)', r'] \1', formatted_text)
    comp_pattern = re.compile(r'\[(.*?)\|(.*?)\]')
    matches = comp_pattern.findall(formatted_text)

    components = []
    relations = []

    for match in matches:
        # print (match)
        comp_str = match[0].strip()  # Text inside the brackets
        tail_span = match[1].strip().split('|')  # Type and relations
        comp_type = tail_span[0].strip()

        if (comp_type in valid_comp_types):
            components.append((comp_type, comp_str))

        if len(tail_span) > 1:
            if (len(tail_span) == 2):
                relations.append((
                    comp_str,
                    tail_span[1].strip()
                ))
            else:
                for i in range(1, len(tail_span)):
                    relations.append((
                    comp_str,
                    tail_span[i].strip()
                ))

    components = set(components)
    relations = set(relations)

    # print("Extracted components:", components)
    # print("Extracted relations:", relations)
    return components, relations

def relation_only_decode_anl(formatted_text):
    # Pattern to match [ text | text ] format
    anl_pattern = re.compile(r'\[(.*?)\]')
    
    # Find all matches of the pattern in the formatted_text
    matches = anl_pattern.findall(formatted_text)

    components = []  # As per the description, components remain unused and empty
    relations = []   # List to store relations extracted from [ head | tail ] patterns

    for match in matches:
        splits = match.split("|")
        if (len(splits) > 1):
            for i in range(1, len(splits)):
                head_span = splits[0].strip()  # Extract and clean the head text
                tail_span = splits[i].strip()  # Extract and clean the tail text
                relations.append((head_span, tail_span))

    # print (relations)

    return components, relations
