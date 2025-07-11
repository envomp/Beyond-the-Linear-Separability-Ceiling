from scripts.conf import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig
from best_PEFT import param_datas

from src.processor import *
from scripts.hf_models import load_phi_3_5_vision, load_pixtral_12B, load_gemma3_4B, resize_images, load_weights, lora_post_dispatch

dataset = "openworld"
if dataset == "hoi":
    dataset_path = HOI_DATASET_PATH
elif dataset == "openworld":
    dataset_path = OPENWORLD_DATASET_PATH
else:
    raise ValueError(f"Dataset not supported: {dataset}")

res = 224
model = "phi"  # phi, pixtral or gemma3_4b
peft = "lora"  # postfix, lora
trainable_prompt_size = 100 if peft == "postfix" else 0
post_dispatch = lora_post_dispatch if peft == "lora" else lambda x: x
sim = True
param_name = None
param_data = torch.load(param_datas[(model, peft, "openworld", sim)], weights_only=True)

if model == "phi":
    (llm, processor) = load_phi_3_5_vision(trainable_prompt_size=trainable_prompt_size, attn="eager", num_crops=1, post_dispatch=post_dispatch)
    epsilon = 1e-8
    if peft == "postfix":
        param_name = "model.vision_embed_tokens.trainable_embeddings"
        prompt_f = lambda x: get_phi_trainable_prompt(x, trainable_tokens=trainable_prompt_size)
    elif "lora" in peft:
        prompt_f = lambda x: get_phi_simple_prompt(x)
    layer_indices = [1, 11, 23, 31]
    layer_titles = ["Layer 2", "Layer 12", "Layer 24", "Layer 32 (last)"]
elif model == "pixtral":
    (llm, processor) = load_pixtral_12B(trainable_prompt_size=trainable_prompt_size, post_dispatch=post_dispatch)
    epsilon = 1e-8
    if peft == "postfix":
        param_name = "language_model.model.embed_tokens.trainable_embeddings"
        prompt_f = lambda x: get_pixtral_trainable_prompt(x, trainable_prompt_size)
    elif "lora" in peft:
        prompt_f = lambda x: get_pixtral_simple_prompt(x)
    layer_indices = [1, 13, 26, 39]
    layer_titles = ["Layer 2", "Layer 14", "Layer 27", "Layer 40 (last)"]
elif model == "gemma3_4b":
    (llm, processor) = load_gemma3_4B(trainable_prompt_size=trainable_prompt_size, post_dispatch=post_dispatch)
    epsilon = 1e-8
    if peft == "postfix":
        prompt_f = lambda x: get_gemma_trainable_prompt(x, trainable_prompt_size)
    elif "lora" in peft:
        prompt_f = lambda x: get_gemma_simple_prompt(x)
    layer_indices = [1, 12, 23, 33]
    layer_titles = ["Layer 2", "Layer 13", "Layer 24", "Layer 34 (last)"]
else:
    raise ValueError(f"Model not supported: {model}")

load_weights(llm, param_data, param_name=param_name)

_, _, test = get_ds(dataset)


def call(urls, prompt):
    images = resize_images([Image.open(url).convert("RGB") for url in urls], longest_edge=res)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(llm.dtype).to("cuda")
    cfg = GenerationConfig(output_attentions=True, output_hidden_states=False, return_dict_in_generate=True)
    outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id, generation_config=cfg)
    print(processor.batch_decode(outputs.sequences[:, inputs['input_ids'].shape[1]:]))
    all_layer_attentions = [attn.detach().cpu().float().numpy() for attn in outputs.attentions[0]]
    return all_layer_attentions


collected_t_attns = []

llm.eval()
cls = 0
item = test[0][1][cls]
urls, _ = get_urls_and_cat_from_item(item, dataset_path=dataset_path)
t_attn = call(urls=urls, prompt=prompt_f(len(urls)) + "answer: cat_")
collected_t_attns.append(t_attn)
attns_all_layers = collected_t_attns[0]

num_layers = len(attns_all_layers)

num_plots = len(layer_indices)
aggregated_attns = []
for idx in layer_indices:
    layer_attn = attns_all_layers[idx]
    agg_attn = np.mean(layer_attn, axis=1)[0, :, :]
    aggregated_attns.append(agg_attn)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

global_min = min(attn.min() for attn in aggregated_attns)
global_max = max(attn.max() for attn in aggregated_attns)

for i in range(num_plots):
    ax = axes[i]
    attention_map = aggregated_attns[i]
    title = layer_titles[i]

    print(f"Plotting {title}: min={attention_map.min():.2e}, max={attention_map.max():.2e}, mean={attention_map.mean():.2e}")

    im = ax.imshow(attention_map + epsilon,
                   cmap='viridis',
                   aspect='auto',
                   norm=mcolors.LogNorm(vmin=max(epsilon, global_min), vmax=global_max))

    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key positions (attended to)")
    ax.set_ylabel("Query positions (attending from)")

for j in range(num_plots, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
fig.savefig(f'{model}_attn_map_{peft}_sim={sim}_item={"pos" if cls==0 else "neg" if cls==1 else "idk"}.png', dpi=300)
