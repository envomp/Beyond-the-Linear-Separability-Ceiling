import random
import string
import json
import os
from transformers.image_processing_utils import BatchFeature
import torch

alphabet = string.ascii_uppercase


def construct_prompt_phi(question: str, choices_with_letters: dict, postfix="") -> str:
    prompt = "<|user|>\n"
    prompt += "Answer this multiple choice question based on the image provided later.\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Here is the image: <|image_1|>\n"
    prompt += "Here are the choices:\n"
    for letter, choice_text in sorted(choices_with_letters.items()):
        prompt += f"{letter}. {choice_text}\n"

    prompt += "\nThe format of your output must be 'Answer: L.' where L is the letter of the correct choice. Do not add any other text.\n"
    prompt += f"<|end|>\n<|assistant|>\n{postfix}"
    return prompt


def process_choices(correct_answer_str, choices_list) -> (dict, str):
    random.shuffle(choices_list)
    choices_with_letters = {}
    correct_letter = None

    for i, choice in enumerate(choices_list):
        letter = alphabet[i]
        choices_with_letters[letter] = choice
        if choice == correct_answer_str:
            correct_letter = letter
    return choices_with_letters, correct_letter


def extract_conclusion(text: str, choices: list) -> str:
    """
    Extracts the most likely answer letter from the model's output.
    This is a more robust version for VQA.
    """
    text_to_search = text
    if "Answer" in text:
        parts = text.rsplit("Answer", 1)
        if len(parts) > 1:
            text_to_search = parts[-1]
    cleaned_text = text_to_search.strip().upper()

    for letter in choices:
        if f"{letter}." in cleaned_text or letter in cleaned_text:
            return letter

    return "INVALID"


def load_gqa_contrastive_training_data(json_file: str, image_dir: str):
    training_data_list = []
    with open(json_file, 'r') as f:
        data = json.load(f)

    for item in data:
        img1_path = os.path.join(image_dir, f"{item['img_id1']}.jpg")
        img2_path = os.path.join(image_dir, f"{item['img_id2']}.jpg")

        min_pair = item['minimize_pair']
        max_pair = item['maximize_pair']
        min_choices_1, min_letter_1 = process_choices(min_pair['vqa_img1']['correct_answer'], min_pair['vqa_img1']['all_answers'])
        min_choices_2, min_letter_2 = process_choices(min_pair['vqa_img2']['correct_answer'], min_pair['vqa_img2']['all_answers'])
        max_choices_1, max_letter_1 = process_choices(min_pair['vqa_img1']['correct_answer'], min_pair['vqa_img1']['all_answers'])
        max_choices_2, max_letter_2 = process_choices(min_pair['vqa_img2']['correct_answer'], min_pair['vqa_img2']['all_answers'])

        training_data_list.append({
            "img1_path": img1_path,
            "img2_path": img2_path,
            "object_concept": item['object_concept'],

            "min_question": min_pair['question'],
            "min_choices_1": min_choices_1,
            "min_answer_1": min_letter_1,
            "min_choices_2": min_choices_2,
            "min_answer_2": min_letter_2,

            "max_question": max_pair['question'],
            "max_choices_1": max_choices_1,
            "max_answer_1": max_letter_1,
            "max_choices_2": max_choices_2,
            "max_answer_2": max_letter_2,
        })

    print(f"Loaded {len(training_data_list)} training pairs.")
    return training_data_list


def load_hoi_prototype_training_data(json_file: str, image_dir: str, split_prefix="train"):
    print(f"Loading prototype training data from {json_file} ({split_prefix}... splits)...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    splits = []

    for split_name, dataset in data.items():
        if not split_name.startswith(split_prefix):
            continue

        training_data_list = []

        for item in dataset:
            meta = item['meta']
            support_sets = item['support_sets']
            formatted_groups = []
            for s_set in support_sets:
                paths = [os.path.join(image_dir, img_rel_path) for img_rel_path in s_set['image_ids']]
                formatted_groups.append({
                    "answer": s_set['prototype_answer'],
                    "paths": paths
                })

            training_data_list.append({
                "question": meta['question'],
                "groups": formatted_groups
            })

        print(f"Loaded {len(training_data_list)} prototype groups for training.")
        splits.append(training_data_list)
    return splits


def pad_left(seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, -len(seq):] = seq
    return padded


def stack_and_pad_inputs(inputs: list[BatchFeature], pad_token_id: int) -> BatchFeature:
    listof_input_ids = [i.input_ids[0] for i in inputs]
    new_input_ids = pad_left(listof_input_ids, pad_token_id=pad_token_id)
    pixel_values_list = [i.pixel_values for i in inputs if i.pixel_values is not None]
    image_sizes_list = [i.image_sizes for i in inputs if i.image_sizes is not None]
    data = {"input_ids": new_input_ids, "attention_mask": (new_input_ids != pad_token_id).long()}
    if pixel_values_list:
        data["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_sizes_list:
        data["image_sizes"] = torch.cat(image_sizes_list, dim=0)
    new_inputs = BatchFeature(data).to("cuda")
    return new_inputs
