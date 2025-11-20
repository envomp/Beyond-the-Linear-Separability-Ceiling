from scripts.conf import *
import json
import random
from collections import defaultdict, deque

# --- CONFIGURATION ---

dataset_config = {
    "train": {
        "files": ["bongard_hoi_train.json"],
        "target_size": 4000,
        "max_reuse": 15,
    },
    "test": {
        "files": [
            "bongard_hoi_val_seen_obj_seen_act.json",
            "bongard_hoi_val_seen_obj_unseen_act.json",
            "bongard_hoi_val_unseen_obj_seen_act.json",
            "bongard_hoi_val_unseen_obj_unseen_act.json",
            "bongard_hoi_test_seen_obj_seen_act.json",
            "bongard_hoi_test_seen_obj_unseen_act.json",
            "bongard_hoi_test_unseen_obj_seen_act.json",
            "bongard_hoi_test_unseen_obj_unseen_act.json"
        ],
        "target_size": 500,
        "max_reuse": 1
    }
}

def parse_concept(concept_str):
    if "++" in concept_str:
        parts = concept_str.split("++")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return None, None

all_data = {}

for group_name, config in dataset_config.items():
    print(f"Processing Group: {group_name} (Target: {config['target_size']} ...")

    # --- Step 1: Aggregate Data from All Files in Group ---
    concept_library = {}
    action_to_concepts = defaultdict(list)
    object_to_concepts = defaultdict(list)

    files_loaded = 0
    for filename in config["files"]:
        try:
            with open("annotations/" + filename, 'r') as f:
                file_data = json.load(f)

            # Process this file's data into the aggregators
            for item in file_data:
                pos_original, neg_original, concept_name = item

                if concept_name not in concept_library:
                    concept_library[concept_name] = []
                    act, obj = parse_concept(concept_name)
                    if act and obj:
                        action_to_concepts[act].append(concept_name)
                        object_to_concepts[obj].append(concept_name)

                img_paths = [x['im_path'].replace("./", "") for x in pos_original]
                concept_library[concept_name].extend(img_paths)

            files_loaded += 1
        except FileNotFoundError:
            print(f"  Warning: annotations/{filename} not found. Skipping.")

    if files_loaded == 0:
        print(f"  Skipping group {group_name}: No files loaded.")
        continue

    # Initial shuffle of image pools
    for c in concept_library:
        random.shuffle(concept_library[c])

    # --- Global Counter for this Group ---
    image_usage_counts = defaultdict(int)

    # Helper: Check validity
    def is_concept_valid(c_name):
        if c_name not in concept_library: return False
        imgs = concept_library[c_name]
        if len(imgs) < 4: return False

        count_avail = 0
        for img in imgs:
            if image_usage_counts[img] < config["max_reuse"]:
                count_avail += 1
            if count_avail >= 4:
                return True
        return False

    def get_all_valid_concepts():
        return [c for c in concept_library if is_concept_valid(c)]

    valid_concepts = get_all_valid_concepts()

    # Helper: Get neighbors
    def get_neighbors(c_name):
        act, obj = parse_concept(c_name)
        neighbors = []
        if act and act in action_to_concepts:
            neighbors.extend([n for n in action_to_concepts[act] if n != c_name])
        if obj and obj in object_to_concepts:
            neighbors.extend([n for n in object_to_concepts[obj] if n != c_name])
        return list(set(neighbors))

    # Clean indices to valid only
    valid_set = set(valid_concepts)
    for k in action_to_concepts:
        action_to_concepts[k] = [c for c in action_to_concepts[k] if c in valid_set]
    for k in object_to_concepts:
        object_to_concepts[k] = [c for c in object_to_concepts[k] if c in valid_set]

    if len(valid_concepts) < 4:
        print(f"  Skipping {group_name}: Not enough concepts found across all files.")
        continue

    # --- Step 3: Generation (BFS Chain) ---
    processed_data = []
    target_size = config["target_size"]
    count = 0
    consecutive_failures = 0

    while count < target_size:
        # Refresh valid concepts periodically
        if consecutive_failures > 10:
            valid_concepts = get_all_valid_concepts()
            consecutive_failures = 0

        if len(valid_concepts) < 4:
            print(f"  Stopped {group_name} at {count}/{target_size}: Data exhausted.")
            break

        # 3.1 Select Anchor
        anchor = random.choice(valid_concepts)
        selected_concepts = [anchor]

        # 3.2 Search for Hard Negatives (BFS)
        search_queue = deque([anchor])

        while len(selected_concepts) < 4 and search_queue:
            current_node = search_queue.popleft()

            neighbors = get_neighbors(current_node)
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if len(selected_concepts) >= 4:
                    break

                if neighbor not in selected_concepts and is_concept_valid(neighbor):
                    selected_concepts.append(neighbor)
                    search_queue.append(neighbor)

        # 3.3 Fallback to Random
        while len(selected_concepts) < 4:
            if len(valid_concepts) < 4: break
            random_cand = random.choice(valid_concepts)
            if random_cand not in selected_concepts:
                selected_concepts.append(random_cand)

        if len(selected_concepts) < 4:
            consecutive_failures += 1
            continue

        random.shuffle(selected_concepts)

        # 3.4 Attempt to fetch images
        sample_images = {}
        success = True
        temp_consumed_ids = []

        for concept in selected_concepts:
            needed = []
            candidates_pool = concept_library[concept]

            for img in candidates_pool:
                if image_usage_counts[img] < config["max_reuse"] and img not in temp_consumed_ids:
                    needed.append(img)
                if len(needed) == 4:
                    break

            if len(needed) < 4:
                success = False
                break

            sample_images[concept] = needed
            temp_consumed_ids.extend(needed)

        if not success:
            consecutive_failures += 1
            continue

        # 3.5 Commit
        for img in temp_consumed_ids:
            image_usage_counts[img] += 1

        support_sets = []
        for concept in selected_concepts:
            support_sets.append({
                "prototype_answer": concept,
                "image_ids": sample_images[concept]
            })

        sample_entry = {
            "meta": {
                "question": f"What interaction best describes what the person is doing?",
                "usage_count_at_gen": len(concept_library[selected_concepts[0]])
            },
            "support_sets": support_sets
        }
        processed_data.append(sample_entry)
        count += 1
        consecutive_failures = 0

    all_data[group_name] = processed_data
    print(f"Finished {group_name}: Generated {len(processed_data)} samples.")

output_file = "../vqa/bongard_hoi_vqa.json"
with open(output_file, 'w') as f:
    json.dump(all_data, f, indent=2)
