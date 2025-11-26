import os
import json
import torch
import random

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

from scripts.hf_models import inference
from processor import process_choices, extract_conclusion, alphabet


def load_hoi_prototype_data(json_file: str, image_dir: str, split_prefix="train"):
    """
    Loads VQA data from the grouped support-set format.
    Flattens N groups * M images into individual samples.
    """
    print(f"Loading VQA prototype data from {json_file} ({split_prefix} split)...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    splits = []
    for split_name, dataset in data.items():
        if not split_name.startswith(split_prefix):
            continue

        dataset = data[split_name]
        flat_vqa_list = []
        for group in tqdm(dataset, desc="Flattening HOI samples"):
            meta_data = group['meta']
            question = meta_data['question']
            support_sets = group['support_sets']

            group_options_pool = [s['prototype_answer'].replace("++", "+").replace("+", " ") for s in support_sets]
            for subset in support_sets:
                correct_ans_text = subset['prototype_answer'].replace("++", "+").replace("+", " ")
                for img_rel_path in subset['image_ids']:
                    image_path = os.path.join(image_dir, img_rel_path)
                    choices_with_letters, correct_letter = process_choices(correct_ans_text, group_options_pool)
                    flat_vqa_list.append({
                        "image_path": image_path,
                        "question": question,
                        "choices_with_letters": choices_with_letters,
                        "object_concept": correct_ans_text,
                        "correct_letter": correct_letter,
                    })
        print(f"Loaded {len(flat_vqa_list)} individual VQA samples for split {split_name}.")
        splits.append(flat_vqa_list)

    return splits


def load_pope_data(split="test"):
    dataset = load_dataset("lmms-lab/POPE", split=split)

    grouped_data = defaultdict(lambda: {'image': None, 'pos': [], 'neg': []})
    for item in dataset:
        img_src = item['image_source']
        if grouped_data[img_src]['image'] is None:
            grouped_data[img_src]['image'] = item['image']
        obj_name = item['question'].replace(" in the image?", "").replace("Is there a ", "")
        if item['answer'].lower() == 'yes':
            grouped_data[img_src]['pos'].append(obj_name)
        else:
            grouped_data[img_src]['neg'].append(obj_name)

    flat_vqa_list = []
    for img_src, data in tqdm(grouped_data.items(), desc="Constructing MC Questions"):
        if len(data['neg']) < 3:
            continue
        distractors = data['neg'][:3]
        target_positives = data['pos'][:3]
        for target_obj in target_positives:
            options_text = [target_obj] + distractors
            random.shuffle(options_text)

            choices_with_letters = {}
            correct_letter = None
            for i, opt in enumerate(options_text):
                letter = alphabet[i]
                choices_with_letters[letter] = opt
                if opt == target_obj:
                    correct_letter = letter

            flat_vqa_list.append({
                "image_path": None,
                "image_object": data['image'].convert('RGB'),
                "question": "Which of the following objects is present in the image?",
                "choices_with_letters": choices_with_letters,
                "correct_letter": correct_letter,
                "meta_img_id": img_src
            })

    print(f"Converted POPE into {len(flat_vqa_list)} Multiple Choice samples.")
    return flat_vqa_list


def load_aokvqa_data(split="validation", max_samples=None): # test has missing answer
    print(f"Loading and formatting A-OKVQA data (config: multiple_choice, split: {split})...")
    dataset = load_dataset("HuggingFaceM4/A-OKVQA", split=split)
    if max_samples:
        dataset = dataset.select(range(max_samples))

    flat_vqa_list = []
    for item in tqdm(dataset, desc="Processing A-OKVQA samples"):
        choices_list = item['choices']
        correct_idx = item['correct_choice_idx']

        choices_with_letters = {}
        correct_letter = None

        for i, choice in enumerate(choices_list):
            letter = alphabet[i]
            choices_with_letters[letter] = choice
            if i == correct_idx:
                correct_letter = letter

        flat_vqa_list.append({
            "image_path": None,
            "image_object": item['image'].convert('RGB'),
            "question": item['question'],
            "choices_with_letters": choices_with_letters,
            "correct_letter": correct_letter,
        })

    print(f"Loaded {len(flat_vqa_list)} individual A-OKVQA samples.")
    return flat_vqa_list

def load_scienceqa_data(split="test", only_with_images=True):
    print(f"Loading and formatting ScienceQA data (split: {split})...")
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    if only_with_images:
        dataset = dataset.filter(lambda x: x['image'] is not None)

    flat_vqa_list = []
    for item in tqdm(dataset, desc="Processing ScienceQA samples"):
        choices_list = item['choices']
        correct_idx = item['answer']

        choices_with_letters = {}
        correct_letter = None
        for i, choice in enumerate(choices_list):
            letter = alphabet[i]
            choices_with_letters[letter] = choice
            if i == correct_idx:
                correct_letter = letter

        img_obj = item['image']
        if img_obj is not None:
            img_obj = img_obj.convert('RGB')

        flat_vqa_list.append({
            "image_path": None,
            "image_object": img_obj,
            "question": item['question'],
            "choices_with_letters": choices_with_letters,
            "correct_letter": correct_letter,
        })

    print(f"Loaded {len(flat_vqa_list)} individual ScienceQA samples.")
    return flat_vqa_list


def run_validation(llm, processor, vqa_dataset, get_full_prompt_f, log=lambda x: print(x)):
    results = defaultdict(int)

    llm.eval()
    with torch.no_grad():
        for i, item in enumerate(vqa_dataset):
            if item.get("image_object"):
                image = item["image_object"]
            else:
                image = Image.open(item["image_path"]).convert('RGB')
            question = item["question"]
            choices_with_letters = item["choices_with_letters"]
            real_answer = item["correct_letter"]
            concept = item.get("object_concept", "N/A")

            valid_letters = list(choices_with_letters.keys())
            full_prompt = get_full_prompt_f(question, choices_with_letters)
            decoded = inference(model=llm, processor=processor, images=[image], prompt=full_prompt, max_tokens=10, skip_special=True, force_cuda=True)
            conc = extract_conclusion(decoded[0], valid_letters)

            if conc == real_answer:
                results["correct"] += 1
            elif conc == "INVALID":
                results["invalid"] += 1
            else:
                results["incorrect"] += 1
            log(f"{i} | expected:'{real_answer}' | got:'{conc}' | concept: '{concept}'\n")
            image.close()

    return results


def run_vqa_baselines(llm, processor, model_str, vqa_dataset, get_full_prompt_f, run_prefix=""):
    RESULTS_DIR = f"result/{model_str}/baselines"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run_filename = f"{run_prefix}results.txt"
    full_file_path = os.path.join(RESULTS_DIR, run_filename)

    with open(full_file_path, 'w', encoding='utf-8', buffering=1) as f_out:
        f_out.write(f"--- {run_prefix} VQA Evaluation ---\n")
        f_out.write(f"  Model: {model_str}\n")
        f_out.write(f"  Total Samples: {len(vqa_dataset)}\n\n")
        f_out.write("---------------------------------------\n")

        results = run_validation(llm, processor, vqa_dataset, get_full_prompt_f, log=f_out.write)

        total_valid = results["correct"] + results["incorrect"]
        accuracy = results["correct"] / max(1, total_valid) * 100
        f_out.write(f"---------------------------------------\n")
        f_out.write(f"Summary:\n")
        f_out.write(f"  Total: {len(vqa_dataset)}\n")
        f_out.write(f"  Correct: {results['correct']}\n")
        f_out.write(f"  Incorrect: {results['incorrect']}\n")
        f_out.write(f"  Invalid/Parse Error: {results['invalid']}\n")
        f_out.write(f"  Processing Error: {results['error']}\n")
        f_out.write(f"  Accuracy (Correct / [Correct + Incorrect]): {accuracy:.2f}%\n")
        f_out.write(f"---------------------------------------\n\n")
