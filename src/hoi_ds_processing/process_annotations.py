from scripts.conf import *
import json
import random

subset_files = [
    ("train", "bongard_hoi_train.json"),
    ("val_seen_obj_seen_act", "bongard_hoi_val_seen_obj_seen_act.json"),
    ("val_seen_obj_unseen_act", "bongard_hoi_val_seen_obj_unseen_act.json"),
    ("val_unseen_obj_seen_act", "bongard_hoi_val_unseen_obj_seen_act.json"),
    ("val_unseen_obj_unseen_act", "bongard_hoi_val_unseen_obj_unseen_act.json"),
    ("test_seen_obj_seen_act", "bongard_hoi_test_seen_obj_seen_act.json"),
    ("test_seen_obj_unseen_act", "bongard_hoi_test_seen_obj_unseen_act.json"),
    ("test_unseen_obj_seen_act", "bongard_hoi_test_unseen_obj_seen_act.json"),
    ("test_unseen_obj_unseen_act", "bongard_hoi_test_unseen_obj_unseen_act.json")
]

all_data = {}

for subset_name, subset_file in subset_files:
    distribution = {}
    with open("annotations/" + subset_file, 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    processed_data = []
    target_size, max_prop = 0, 0
    if subset_name == "train":
        target_size = 4000
        max_prop = 50
    if "val_" in subset_name:
        target_size = 100
        max_prop = 5
    if "test_" in subset_name:
        target_size = 200
        max_prop = 8
    i = -1
    while len(processed_data) < target_size and i < len(data):
        i += 1
        pos_original, neg_original, concept = data[i]
        if concept not in distribution:
            distribution[concept] = 0

        if distribution[concept] > max_prop:
            continue  # balanced datasets
        distribution[concept] += 1

        pos = list(x['im_path'].replace("./", "") for x in pos_original)
        neg = list(x['im_path'].replace("./", "") for x in neg_original)
        random.shuffle(pos)
        random.shuffle(neg)

        pos_test_idx = random.randrange(len(pos))
        pos_test_image = pos.pop(pos_test_idx)

        neg_test_idx = random.randrange(len(neg))
        neg_test_image = neg.pop(neg_test_idx)

        sample_pos_test = {
            "test_id": f"{concept}_pos_test",
            "uid": concept,
            "imagefiles": {
                "cat_1": neg,
                "cat_2": pos
            },
            "testfiles": {
                "category": "cat_2",
                "testimage": pos_test_image
            },
            "commonSense": "0",
            "concept": concept,
            "caption": f"{concept}."
        }
        processed_data.append(sample_pos_test)

        sample_neg_test = {
            "test_id": f"{concept}_neg_test",
            "uid": concept,
            "imagefiles": {
                "cat_1": neg,
                "cat_2": pos
            },
            "testfiles": {
                "category": "cat_1",
                "testimage": neg_test_image
            },
            "commonSense": "0",
            "concept": concept,
            "caption": f"{concept}."
        }
        processed_data.append(sample_neg_test)

    all_data[subset_name] = processed_data
    print(f"distribution of {subset_name}: {sum(distribution.values())} {distribution}")

with open("../bongard_hoi.json", 'w') as f:
    json.dump(all_data, f, indent=4)
