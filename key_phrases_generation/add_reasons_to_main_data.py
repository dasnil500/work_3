import json
import re

json_file_path = "/workspace/saradhi_pg/Nil/llama-experiment/data/aaec_para/aaec_para_train.json"
text_file_path = "/workspace/saradhi_pg/Nil/llama-experiment/5.key_phrases_generation/key_phrases_train_aaec_para.txt"

with open(json_file_path, "r") as f:
    json_data = json.load(f)

with open(text_file_path, "r") as f:
    text_data = f.read()

text_instances = text_data.split("--------------------------------------------------------------------------------------------------------")

reason_pattern = r"Reason:\s*([\s\S]*?)(?=\n[-]+|\Z)"

for item in text_instances:

    lines = item.split("\n")

    AC_1 = lines[3][7:-1]
    AC_2 = lines[4][7:-1]

    # print (f"\nHead: {AC_1}\nTail: {AC_2}")
    
    reason_match = re.search(reason_pattern, item)

    reasons = ""

    if reason_match:
        reason = reason_match.group(1).strip()
        all_reasons = reason.split("\n")
        # print (f"Reasons are:")

        for r in all_reasons:
            r = r.strip()

            if (r.startswith("-")):

                r = r.strip("- ")
                reasons = reasons + " " + r
            else:
                print ("No Valid Reason.")
    else:
        print("No reason found.")

    for instance in json_data:
        for relation in instance['relations']:
            head_component = instance['components'][relation['head']]['span']
            tail_component = instance['components'][relation['tail']]['span']

            # Match head and tail with AC_1 and AC_2
            if AC_1 == head_component and AC_2 == tail_component:
                relation['reasons'] = reasons.strip()
                print(f"Added Reasons: {reasons}")
                break

with open("updated_aaec_para_test.json", "w") as f:
    json.dump(json_data, f, indent=4)
