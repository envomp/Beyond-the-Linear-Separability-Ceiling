import os
import json
import torch

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset

from scripts.hf_models import inference
from processor import process_choices, extract_conclusion, alphabet


def load_gqa_data(json_file: str, image_dir: str):
    print(f"Loading and flattening VQA data from {json_file}...")
    flat_vqa_list = []

    with open(json_file, 'r') as f:
        data = json.load(f)

    for item in tqdm(data, desc="Processing GQA pairs"):
        pairs_to_process = [
            (item['img_id1'], item['minimize_pair']['question'], item['minimize_pair']['vqa_img1'], 'minimize'),
            (item['img_id2'], item['minimize_pair']['question'], item['minimize_pair']['vqa_img2'], 'minimize'),
            (item['img_id1'], item['maximize_pair']['question'], item['maximize_pair']['vqa_img1'], 'maximize'),
            (item['img_id2'], item['maximize_pair']['question'], item['maximize_pair']['vqa_img2'], 'maximize')
        ]

        for img_id, question, vqa_block, pair_type in pairs_to_process:
            image_path = os.path.join(image_dir, f"{img_id}.jpg")
            choices_with_letters, correct_letter = process_choices(vqa_block)

            flat_vqa_list.append({
                "image_path": image_path,
                "question": question,
                "choices_with_letters": choices_with_letters,
                "correct_letter": correct_letter,
                "object_concept": item['object_concept'],
                "pair_type": pair_type
            })

    print(f"Loaded {len(flat_vqa_list)} individual VQA samples.")
    return flat_vqa_list


def load_aokvqa_data(split="validation", max_samples=None):
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
