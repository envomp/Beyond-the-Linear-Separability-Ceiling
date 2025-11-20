from scripts.conf import *
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers.image_processing_utils import BatchFeature

from scripts.hf_models import load_phi_3_5_vision, lora_post_dispatch, resize_images, find_image_token_ranges_phi
from processor import construct_prompt_phi, load_gqa_contrastive_training_data, stack_and_pad_inputs
from baseline import load_gqa_contrastive_data, run_validation

seed = random.randint(1, 10000)
model = "phi"
train_json = "gqa_contrastive_pairs_train.json"
eval_json = "gqa_contrastive_pairs_eval.json"
wc = 0.4
num_epochs = 1
batch_size = 8
eps = 1e-8
contrastive_temperature = 0.07
learning_rate = 0.0001
resolution = 224

exec(open('configurator.py').read())

print(f"Fixing seed to: {seed}")
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def validate_model(llm, processor, vqa_dataset, get_full_prompt_f):
    results = run_validation(llm, processor, vqa_dataset, get_full_prompt_f)
    accuracy = (results["correct"] / max(1, results["correct"] + results["incorrect"])) * 100
    print(f"Validation Summary:")
    print(f"  Correct: {results['correct']}, Incorrect: {results['incorrect']}, Invalid: {results['invalid']}")
    print(f"  Accuracy: {accuracy:.2f}%")
    return accuracy, results['correct']

post_dispatch = lambda x: lora_post_dispatch(x, ignore_vision=True)  # no benefit in training vision encoder
llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch, do_checkpoint=True)
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
pad_token_id = processor.tokenizer.pad_token_id
train_data = load_gqa_contrastive_training_data(train_json, GQA_IMAGE_DIR)
val_data = load_gqa_contrastive_data(eval_json, GQA_IMAGE_DIR)

trainable_params = [p for n, p in llm.named_parameters() if p.requires_grad]
print(f"Training {len(trainable_params)} parameters.")
optimizer = optim.AdamW(trainable_params, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

num_batches_per_epoch = len(train_data) // batch_size
for epoch in range(num_epochs):
    llm.train()
    random.shuffle(train_data)
    total_epoch_loss = 0
    total_epoch_vqa_loss = 0
    total_epoch_contrast_loss = 0

    for i in range(0, len(train_data), batch_size):
        batch_items = train_data[i: i + batch_size]
        optimizer.zero_grad()

        listof_inputs_full: list[BatchFeature] = []
        listof_prompt_lens: list[int] = []
        for item in batch_items:
            img1 = Image.open(item['img1_path']).convert("RGB")
            img2 = Image.open(item['img2_path']).convert("RGB")
            if resolution:
                img1, img2 = resize_images([img1, img2], longest_edge=resolution)


            def construct_vqa_prompt_and_labels(question: str, choices_with_letters: dict, correct_letter: str):
                prompt_text = construct_prompt_phi(question, choices_with_letters)
                answer_text = f"Answer: {correct_letter}."
                full_text = prompt_text + answer_text
                return full_text, prompt_text, answer_text


            samples_to_process = [
                (item['min_question'], item['min_choices_1'], item['min_answer_1'], img1),
                (item['min_question'], item['min_choices_2'], item['min_answer_2'], img2),
                (item['max_question'], item['max_choices_1'], item['max_answer_1'], img1),
                (item['max_question'], item['max_choices_2'], item['max_answer_2'], img2),
            ]

            for (q, c, a, img) in samples_to_process:
                full_text, prompt_text, answer_text = construct_vqa_prompt_and_labels(q, c, a)
                inputs_full = processor(text=full_text, images=[img], return_tensors="pt")
                answer_tokens = processor(text=answer_text, return_tensors="pt").input_ids
                full_len = inputs_full.input_ids.shape[1]
                answer_len = answer_tokens.shape[1]
                prompt_len = full_len - answer_len
                listof_inputs_full.append(inputs_full)
                listof_prompt_lens.append(prompt_len)

        inputs = stack_and_pad_inputs(listof_inputs_full, pad_token_id)
        inputs = inputs.to(llm.dtype)
        padded_labels = inputs.input_ids.clone()
        max_len = padded_labels.shape[1]
        for j in range(len(listof_inputs_full)):
            prompt_len = listof_prompt_lens[j]
            full_seq_len = listof_inputs_full[j].input_ids.shape[1]
            seq_start_index = max_len - full_seq_len
            end_index_of_prompt = seq_start_index + prompt_len
            padded_labels[j, :end_index_of_prompt] = -100

        out = llm(**inputs, output_hidden_states=True, use_cache=False)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = padded_labels[..., 1:].contiguous()
        vqa_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # --- Calculate contrastive loss ---
        hidden_states = out.hidden_states[-1]  # Shape: [batch_size*4, seq_len, hidden_dim]
        all_image_ranges = find_image_token_ranges_phi(inputs)

        reps_list = []
        for j in range(hidden_states.shape[0]):
            sample_hidden_state = hidden_states[j]  # Shape: [seq_len, hidden_dim]
            sample_image_ranges_dict = all_image_ranges[j]
            (start_idx, end_idx) = list(sample_image_ranges_dict.values())[0]
            image_token_hidden_states = sample_hidden_state[start_idx: end_idx + 1, :]
            visual_rep = torch.mean(image_token_hidden_states, dim=0)  # Shape: [hidden_dim]
            reps_list.append(visual_rep)

        reps = torch.stack(reps_list)  # Shape: [batch_size*4, hidden_dim]
        rep_min1s, rep_min2s = reps[0::4], reps[1::4]  # Shape: [batch_size, hidden_dim]
        rep_max1s, rep_max2s = reps[2::4], reps[3::4]  # Shape: [batch_size, hidden_dim]

        rep_min1_norm, rep_min2_norm = F.normalize(rep_min1s, p=2, dim=1, eps=eps), F.normalize(rep_min2s, p=2, dim=1, eps=eps)
        rep_max1_norm, rep_max2_norm = F.normalize(rep_max1s, p=2, dim=1, eps=eps), F.normalize(rep_max2s, p=2, dim=1, eps=eps)
        sim_positive, sim_negative = (rep_min1_norm * rep_min2_norm).sum(dim=1), (rep_max1_norm * rep_max2_norm).sum(dim=1)

        contrastive_logits = torch.stack([sim_positive, sim_negative], dim=1) / contrastive_temperature
        contrastive_target = torch.zeros(rep_min1_norm.shape[0], dtype=torch.long, device=reps.device)
        total_contrast_loss = loss_fn(contrastive_logits, contrastive_target)

        loss = vqa_loss + (wc * total_contrast_loss)

        loss.backward()
        optimizer.step()

        total_epoch_loss += loss.item()
        total_epoch_vqa_loss += vqa_loss.item()
        total_epoch_contrast_loss += total_contrast_loss.item()

        if (i // batch_size) % 20 == 0:
            print(f"  Batch {i // batch_size}/{num_batches_per_epoch}: Batch Loss: {loss.item():.4f} (VQA: {vqa_loss.item():.4f}, Contrast: {total_contrast_loss.item():.4f}), wc: {wc:.3f}, wn: {1:.3f}")

    avg_loss = total_epoch_loss / num_batches_per_epoch
    avg_vqa_loss = total_epoch_vqa_loss / num_batches_per_epoch
    avg_contrast_loss = total_epoch_contrast_loss / num_batches_per_epoch
    print(f"Epoch {epoch + 1} Complete. Avg Loss: {avg_loss:.4f} (Avg VQA: {avg_vqa_loss:.4f}, Avg Contrast: {avg_contrast_loss:.4f})")

    val_accuracy, val_correct = validate_model(llm, processor, val_data, construct_prompt_phi)

    state_tensors = {}
    for name, param in llm.named_parameters():
        if param.requires_grad:
            state_tensors[name] = param.data.clone().cpu()

    filename = f"sim_gqa_{model}_lora_c_{wc}_e_{epoch}_acc_{val_correct}_seed_{seed}.pt"

    save_path = os.path.join(PEFT_PATH, filename)
    torch.save(state_tensors, save_path)
    print(f"Saved LoRA weights to '{save_path}'")
