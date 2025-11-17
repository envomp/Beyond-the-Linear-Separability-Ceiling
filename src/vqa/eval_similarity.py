from scripts.conf import *
from PIL import Image
import torch
import torch.nn.functional as F
from transformers.image_processing_utils import BatchFeature

from scripts.hf_models import load_phi_3_5_vision, lora_post_dispatch, resize_images, find_image_token_ranges_phi, load_weights
from processor import construct_prompt_phi, load_gqa_training_data, stack_and_pad_inputs

loras = {
    ("phi", "baseline"): None,
    ("phi", "gqa_c_0.4"): "sim_gqa_phi_lora_c_0.4_e_0_acc_8819_seed_941.pt",
    ("phi", "gqa_c_0.1"): "sim_gqa_phi_lora_c_0.1_e_0_acc_9873_seed_952.pt",
    ("phi", "gqa_c_0.01"): "sim_gqa_phi_lora_c_0.01_e_0_acc_9728_seed_1012.pt",
    ("phi", "gqa_nt"): "sim_gqa_phi_lora_c_0.0_e_0_acc_9847_seed_7894.pt",
}

resolution = 224
eps = 1e-8


def construct_vqa_prompt_and_labels(question: str, choices_with_letters: dict, correct_letter: str):
    prompt_text = construct_prompt_phi(question, choices_with_letters)
    answer_text = f"Answer: {correct_letter}."
    full_text = prompt_text + answer_text
    return full_text, prompt_text, answer_text


for model, method in loras:
    post_dispatch = lambda x: lora_post_dispatch(x, ignore_vision=True)
    llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch if method != "baseline" else lambda x: x)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    pad_token_id = processor.tokenizer.pad_token_id
    if method != "baseline":
        param_data = torch.load(PEFT_PATH + loras[(model, method)], weights_only=True)
        load_weights(llm.eval(), param_data, no_vision=True)
    prompt = lambda x, y: construct_prompt_phi(x, y, postfix="Answer:")
    vqa_dataset = load_gqa_training_data("gqa_contrastive_pairs_eval.json", GQA_IMAGE_DIR)

    llm.eval()
    all_minimize_sims = []
    all_maximize_sims = []

    with torch.no_grad():
        for item in vqa_dataset[:100]:
            listof_inputs_full: list[BatchFeature] = []
            img1 = Image.open(item['img1_path']).convert("RGB")
            img2 = Image.open(item['img2_path']).convert("RGB")
            if resolution:
                img1, img2 = resize_images([img1, img2], longest_edge=resolution)
            samples_to_process = [
                (item['min_question'], item['min_choices_1'], item['min_answer_1'], img1),
                (item['min_question'], item['min_choices_2'], item['min_answer_2'], img2),
                (item['max_question'], item['max_choices_1'], item['max_answer_1'], img1),
                (item['max_question'], item['max_choices_2'], item['max_answer_2'], img2),
            ]
            for (q, c, a, img) in samples_to_process:
                full_text, _, _ = construct_vqa_prompt_and_labels(q, c, a)
                inputs_full = processor(text=full_text, images=[img], return_tensors="pt")
                listof_inputs_full.append(inputs_full)

            inputs = stack_and_pad_inputs(listof_inputs_full, pad_token_id)
            inputs = inputs.to(llm.dtype)
            out = llm(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states[-1]
            all_image_ranges = find_image_token_ranges_phi(inputs)

            reps_list = []
            for j in range(hidden_states.shape[0]):
                sample_hidden_state = hidden_states[j]
                sample_image_ranges_dict = all_image_ranges[j]
                (start_idx, end_idx) = list(sample_image_ranges_dict.values())[0]
                image_token_hidden_states = sample_hidden_state[start_idx: end_idx + 1, :]
                visual_rep = torch.mean(image_token_hidden_states, dim=0)
                reps_list.append(visual_rep)
            reps = torch.stack(reps_list)

            rep_min1s, rep_min2s = reps[0::4], reps[1::4]
            rep_max1s, rep_max2s = reps[2::4], reps[3::4]

            rep_min1_norm, rep_min2_norm = F.normalize(rep_min1s, p=2, dim=1, eps=eps), F.normalize(rep_min2s, p=2, dim=1, eps=eps)
            rep_max1_norm, rep_max2_norm = F.normalize(rep_max1s, p=2, dim=1, eps=eps), F.normalize(rep_max2s, p=2, dim=1, eps=eps)
            sim_minimize_batch = (rep_min1_norm * rep_min2_norm).sum(dim=1)
            sim_maximize_batch = (rep_max1_norm * rep_max2_norm).sum(dim=1)

            all_minimize_sims.extend(sim_minimize_batch.cpu().tolist())
            all_maximize_sims.extend(sim_maximize_batch.cpu().tolist())

    avg_min_sim = sum(all_minimize_sims) / len(all_minimize_sims)
    avg_max_sim = sum(all_maximize_sims) / len(all_maximize_sims)

    print(f"\n--- Similarity Evaluation Results for {method} ---")
    print(f"Total pairs evaluated: {len(all_minimize_sims)}")
    print(f"  average 'minimize' (positive) similarity: {avg_min_sim:.4f}")
    print(f"  average 'maximize' (negative) similarity: {avg_max_sim:.4f}")
    print("---------------------------------------")
    print(f"  Difference (Minimize - Maximize): {avg_min_sim - avg_max_sim: .4f}")
    print("\n")
    print("Expected: 'Minimize' similarity > 'Maximize' similarity.")
