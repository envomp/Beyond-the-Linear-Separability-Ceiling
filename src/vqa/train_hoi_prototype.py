from scripts.conf import *
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from transformers.image_processing_utils import BatchFeature

from scripts.hf_models import load_phi_3_5_vision, lora_post_dispatch, resize_images, find_image_token_ranges_phi
from processor import construct_prompt_phi, load_hoi_prototype_training_data, stack_and_pad_inputs, process_choices
from baseline import run_validation, load_hoi_prototype_data
from loss import PrototypicalBatchLoss

seed = random.randint(1, 10000)
model = "phi"
hoi_json = "bongard_hoi_vqa.json"
wc = 0.1
num_epochs = 1
batch_size = 2  # each "sample" contains 4x4 images, so effective batch size is batch_size * 16
eps = 1e-8
contrastive_temperature = 0.07
learning_rate = 0.0001
resolution = 336

exec(open('configurator.py').read())

print(f"Fixing seed to: {seed}")
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def validate_model(llm, processor, vqa_dataset, get_full_prompt_f):
    results = run_validation(llm, processor, vqa_dataset, get_full_prompt_f)
    accuracy = (results["correct"] / max(1, results["correct"] + results["incorrect"])) * 100
    print(f"Validation summary:")
    print(f"  correct: {results['correct']}, incorrect: {results['incorrect']}, invalid: {results['invalid']}")
    print(f"  accuracy: {accuracy:.2f}%")
    return accuracy, results['correct']


post_dispatch = lambda x: lora_post_dispatch(x, ignore_vision=True)  # no benefit in training vision encoder
llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch, do_checkpoint=True)
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
pad_token_id = processor.tokenizer.pad_token_id
train_data = load_hoi_prototype_training_data(hoi_json, HOI_DATASET_PATH, split_prefix="train")[0]
val_data = load_hoi_prototype_data(hoi_json, HOI_DATASET_PATH, split_prefix="test")[0]

trainable_params = [p for n, p in llm.named_parameters() if p.requires_grad]
print(f"Training {len(trainable_params)} parameters.")
optimizer = optim.AdamW(trainable_params, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
proto_loss_fn = PrototypicalBatchLoss(temperature=contrastive_temperature)

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
        batch_group_labels = []
        global_group_counter = 0
        images_to_process_flat = []

        for item in batch_items:
            question = item['question']
            groups = item['groups']
            options_pool = [g['answer'] for g in groups]

            for group in groups:
                current_group_id = global_group_counter
                global_group_counter += 1

                correct_ans = group['answer']
                for path in group['paths']:
                    img = Image.open(path).convert("RGB")
                    choices_with_letters, correct_letter = process_choices(correct_ans, options_pool)
                    prompt_text = construct_prompt_phi(question, choices_with_letters)
                    answer_text = f"Answer: {correct_letter}."
                    full_text = prompt_text + answer_text
                    images_to_process_flat.append(img)
                    batch_group_labels.append(current_group_id)
                    inputs_full = processor(text=full_text, images=[img], return_tensors="pt")
                    answer_tokens = processor(text=answer_text, return_tensors="pt").input_ids
                    full_len = inputs_full.input_ids.shape[1]
                    answer_len = answer_tokens.shape[1]
                    prompt_len = full_len - answer_len
                    listof_inputs_full.append(inputs_full)
                    listof_prompt_lens.append(prompt_len)

        if resolution:
            images_to_process_flat = resize_images(images_to_process_flat, longest_edge=resolution)
        inputs = stack_and_pad_inputs(listof_inputs_full, pad_token_id)
        inputs = inputs.to(llm.dtype)
        padded_labels = inputs.input_ids.clone()
        padded_labels[padded_labels == pad_token_id] = -100
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

        hidden_states = out.hidden_states[-1]
        all_image_ranges = find_image_token_ranges_phi(inputs)
        all_reps = []
        for j in range(hidden_states.shape[0]):
            sample_hidden_state = hidden_states[j]
            sample_image_ranges_dict = all_image_ranges[j]
            (start_idx, end_idx) = list(sample_image_ranges_dict.values())[0]
            image_token_hidden_states = sample_hidden_state[start_idx: end_idx + 1, :]
            visual_rep = torch.mean(image_token_hidden_states, dim=0)
            all_reps.append(visual_rep)

        features = torch.stack(all_reps) # [batch_size, hidden_dim]
        batch_group_labels_tensor = torch.tensor(batch_group_labels).to(features.device)
        contrast_loss = proto_loss_fn(features, batch_group_labels_tensor)

        loss = vqa_loss + (wc * contrast_loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), max_norm=3.0)
        optimizer.step()

        total_epoch_loss += loss.item()
        total_epoch_vqa_loss += vqa_loss.item()
        total_epoch_contrast_loss += contrast_loss.item()

        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i // batch_size}/{num_batches_per_epoch}: batch loss: {loss.item():.4f} (VQA: {vqa_loss.item():.4f}, contrast: {contrast_loss.item():.4f})")

    # --- END EPOCH ---
    avg_loss = total_epoch_loss / num_batches_per_epoch
    avg_vqa_loss = total_epoch_vqa_loss / num_batches_per_epoch
    avg_contrast_loss = total_epoch_contrast_loss / num_batches_per_epoch
    print(f"Epoch {epoch + 1} complete. avg loss: {avg_loss:.4f} (avg VQA: {avg_vqa_loss:.4f}, avg contrast: {avg_contrast_loss:.4f})")

    val_accuracy, val_correct = validate_model(llm, processor, val_data, construct_prompt_phi)

    state_tensors = {}
    for name, param in llm.named_parameters():
        if param.requires_grad:
            state_tensors[name] = param.data.clone().cpu()

    filename = f"sim_prototype_hoi_{model}_lora_wc_{wc}_epoch_{epoch}_acc_{val_correct}_seed_{seed}.pt"

    save_path = os.path.join(PEFT_PATH, filename)
    torch.save(state_tensors, save_path)
    print(f"Saved LoRA weights to '{save_path}'")
