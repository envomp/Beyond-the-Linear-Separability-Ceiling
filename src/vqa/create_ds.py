"""
This script mines the GQA dataset to find contrastive training tuples
based on the logic:
(Img1, Img2) are semantically related (e.g., both contain a 'hydrant').
Q_same is a question where (Img1, Q_same) and (Img2, Q_same) have the SAME answer.
Q_diff is a question where (Img1, Q_diff) and (Img2, Q_diff) have DIFFERENT answers.

GQA scene graph and question files can be downloaded from:
https://cs.stanford.edu/people/dorarad/gqa/download.html
"""

import json
import random
from collections import defaultdict
from tqdm import tqdm

create_train_ds = True

if create_train_ds:
    SCENE_GRAPH_FILE = "/home/e/Downloads/sceneGraphs/train_sceneGraphs.json"
    QUESTIONS_FILE = "/home/e/Downloads/questions1.2/train_balanced_questions.json"
    OUTPUT_FILE = "gqa_contrastive_pairs_train.json"
    MAX_APPEARANCES_PER_IMAGE = 2
    MAX_PAIRS_PER_OBJECT = 200
else:
    SCENE_GRAPH_FILE = "/home/e/Downloads/sceneGraphs/val_sceneGraphs.json"
    QUESTIONS_FILE = "/home/e/Downloads/questions1.2/val_balanced_questions.json"
    OUTPUT_FILE = "gqa_contrastive_pairs_eval.json"
    MAX_APPEARANCES_PER_IMAGE = 2
    MAX_PAIRS_PER_OBJECT = 25
# Total number of answers for VQA (1 correct + 3 distractors)
NUM_TOTAL_ANSWERS = 4


def build_indices(questions_data, scene_graphs):
    """
    Builds four key indices:
    1. object_to_images: Maps an object name to a set of imageIds.
    2. image_to_questions: Maps an imageId to a list of {question, answer} dicts.
    3. question_to_local_group: Maps a question string to its local group ID.
    4. local_group_to_answers_pool: Maps a local group ID to a set of all its
       answers from across the dataset (for sampling hard distractors).
    """
    print("Building index: imageId -> questions...")
    image_to_questions = defaultdict(list)
    local_group_to_answers_pool = defaultdict(set)
    question_to_local_group = {}

    for q_data in tqdm(questions_data.values(), desc="Indexing questions"):
        question_text = q_data['question']
        answer = q_data['answer']
        if 'local' in q_data['groups']:
            local_group = q_data['groups']['local']
        else:
            # Use global group as a fallback if local is not present
            local_group = q_data['groups'].get('global', question_text)
        image_to_questions[q_data['imageId']].append({'q': question_text, 'a': answer})
        local_group_to_answers_pool[local_group].add(answer)
        if question_text not in question_to_local_group:
            question_to_local_group[question_text] = local_group

    print("Building index: object_name -> imageIds...")
    object_to_images = defaultdict(set)
    for img_id, scene_graph in tqdm(scene_graphs.items(), desc="Indexing objects"):
        if img_id not in image_to_questions:
            continue
        if 'objects' not in scene_graph:
            continue
        for obj_data in scene_graph['objects'].values():
            obj_name = obj_data['name']
            object_to_images[obj_name].add(img_id)

    print(f"Found {len(local_group_to_answers_pool)} unique question groups.")
    return object_to_images, image_to_questions, question_to_local_group, local_group_to_answers_pool


def get_vqa_answers(correct_answer, local_group_id, local_group_to_answers_pool):
    """
    Generates a list of VQA answers (1 correct, num_total-1 hard distractors).
    Distractors are sampled from other answers given to the *same local group*.
    """
    num_distractors_needed = NUM_TOTAL_ANSWERS - 1
    all_answers_for_group = local_group_to_answers_pool.get(local_group_id, set())
    distractor_pool_list = list(all_answers_for_group - {correct_answer})
    num_distractors_available = len(distractor_pool_list)
    num_to_sample = min(num_distractors_needed, num_distractors_available)

    if num_to_sample > 0:
        distractors = random.sample(distractor_pool_list, num_to_sample)
    else:
        distractors = []

    final_list = distractors + [correct_answer]
    random.shuffle(final_list)
    return final_list


def find_contrastive_pairs(object_to_images, image_to_questions, question_to_local_group, local_group_to_answers_pool):
    """ The core logic. Finds the (Img1, Img2, Q_same, Q_diff) tuples. """
    print("Mining for contrastive training pairs...")
    training_pairs = []
    image_usage_count = defaultdict(int)

    for obj_name, image_ids in tqdm(object_to_images.items(), desc="Mining pairs"):
        if len(image_ids) < 2:
            continue

        image_list = list(image_ids)
        random.shuffle(image_list)

        object_pair_count = 0

        for i in range(len(image_list)):
            if object_pair_count >= MAX_PAIRS_PER_OBJECT:
                break

            img_id1 = image_list[i]
            if image_usage_count[img_id1] >= MAX_APPEARANCES_PER_IMAGE:
                continue

            for j in range(i + 1, len(image_list)):
                if object_pair_count >= MAX_PAIRS_PER_OBJECT:
                    break

                img_id2 = image_list[j]

                q_list1 = image_to_questions.get(img_id1, [])
                q_list2 = image_to_questions.get(img_id2, [])

                if not q_list1 or not q_list2:
                    continue

                q_map1 = {item['q']: item['a'] for item in q_list1}
                q_map2 = {item['q']: item['a'] for item in q_list2}

                common_q_texts = set(q_map1.keys()) & set(q_map2.keys())

                if len(common_q_texts) < 2:
                    continue

                minimize_example = None
                maximize_example = None

                common_q_list = list(common_q_texts)
                random.shuffle(common_q_list)

                for q_text in common_q_list:
                    local_group = question_to_local_group.get(q_text)
                    if not local_group:
                        local_group = q_text

                    ans1 = q_map1[q_text]
                    ans2 = q_map2[q_text]

                    if ans1 == ans2 and not minimize_example:
                        ans = ans1
                        vqa1_answers = get_vqa_answers(ans, local_group, local_group_to_answers_pool)
                        vqa2_answers = get_vqa_answers(ans, local_group, local_group_to_answers_pool)
                        if len(vqa1_answers) < 2 or len(vqa2_answers) < 2:
                            continue
                        minimize_example = {
                            'question': q_text,
                            'vqa_img1': { 'correct_answer': ans, 'all_answers': vqa1_answers },
                            'vqa_img2': { 'correct_answer': ans, 'all_answers': vqa2_answers }
                        }

                    if ans1 != ans2 and not maximize_example:
                        vqa1_answers = get_vqa_answers(ans1, local_group, local_group_to_answers_pool)
                        vqa2_answers = get_vqa_answers(ans2, local_group, local_group_to_answers_pool)
                        if len(vqa1_answers) < 2 or len(vqa2_answers) < 2:
                            continue
                        maximize_example = {
                            'question': q_text,
                            'vqa_img1': { 'correct_answer': ans1, 'all_answers': vqa1_answers },
                            'vqa_img2': { 'correct_answer': ans2, 'all_answers': vqa2_answers }
                        }

                    if minimize_example and maximize_example:
                        training_pairs.append({
                            'object_concept': obj_name,
                            'img_id1': img_id1,
                            'img_id2': img_id2,
                            'minimize_pair': minimize_example,
                            'maximize_pair': maximize_example
                        })
                        image_usage_count[img_id1] += 1
                        image_usage_count[img_id2] += 1
                        object_pair_count += 1
                        break

    return training_pairs


with open(SCENE_GRAPH_FILE, 'r') as f:
    scene_graphs = json.load(f)

with open(QUESTIONS_FILE, 'r') as f:
    questions_data = json.load(f)

print("Building indices...")
object_to_images, image_to_questions, question_to_local_group, local_group_to_answers_pool = build_indices(questions_data, scene_graphs)
print("Indices built.")

training_tuples = find_contrastive_pairs(
    object_to_images,
    image_to_questions,
    question_to_local_group,
    local_group_to_answers_pool
)


print(f"\n--- Mining Complete ---")
print(f"Found {len(training_tuples)} total contrastive training tuples.")

if training_tuples:
    print(f"Saving tuples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(training_tuples, f, indent=2)

    print("\n--- Example Tuple (New Format) ---")
    print(json.dumps(random.choice(training_tuples), indent=2))
    print("----------------------------------")
    print("Script finished successfully.")
else:
    print("No training tuples found. This might be an issue or the data split is small.")
