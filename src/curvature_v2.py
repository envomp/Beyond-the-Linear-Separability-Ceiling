from scripts.conf import *
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from transformers import GenerationConfig

from scripts.hf_models import load_phi_3_5_vision, load_weights, lora_post_dispatch, resize_images
from scripts.hf_models import find_image_token_ranges_phi
from processor import get_ds, get_urls_and_cat_from_item, get_phi_simple_prompt, interleaved_prompt
from tqdm import tqdm

param_datas = {
    ("phi", "lora", "hoi", False): "sim_hoi_phi_lora_c_0.0_e_2_t_0_acc_74_81_77_82_seed_9712.pt",
    ("phi", "lora", "hoi", True): "sim_hoi_phi_lora_c_0.4_e_2_t_0_acc_79_80_81_82_seed_4645.pt",
}

def get_cat2_dataset_means(llm, processor, dataset, dataset_path):
    """
    Iterates through the dataset, extracts Cat_2 (Positive) images,
    and computes their mean embeddings.
    """
    prompt_fn = lambda x: get_phi_simple_prompt(x, base_prompt=interleaved_prompt)
    ranges = find_image_token_ranges_phi
    prototypes = []

    for item in tqdm(dataset):
        urls, _ = get_urls_and_cat_from_item(item, dataset_path)
        images = resize_images([Image.open(url).convert("RGB") for url in urls], longest_edge=224)
        inputs = processor(images=images, text=prompt_fn(len(images)), return_tensors="pt").to(llm.dtype).to("cuda")
        cfg = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True, use_cache=False)
        with torch.no_grad():
            outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1,
                                   eos_token_id=processor.tokenizer.eos_token_id, generation_config=cfg)
        layer_hidden_states = outputs["hidden_states"][0][-1][0]
        indexes = ranges(inputs)

        current_item_vectors = []
        for img_idx in range(1, 7):
            start, end = indexes[0][img_idx]
            tokens = layer_hidden_states[start:end, :].float().cpu()
            img_mean = tokens.mean(dim=0)
            current_item_vectors.append(img_mean)
        item_prototype = torch.stack(current_item_vectors).mean(dim=0).numpy()
        prototypes.append(item_prototype)

    return np.array(prototypes)

# model_name = "phi"
# learnable_embedding_location = "lora"
# dataset_name = "hoi"
# sim = False
#
# print(f"--- Visualizing 3D Prototype Manifold: Sim={sim} ---")
# param_key = (model_name, learnable_embedding_location, dataset_name, sim)
# param_file = param_datas.get(param_key)
# if param_file:
#     param_file = PEFT_PATH + param_file
#     param_data = torch.load(param_file, weights_only=True)
#     post_dispatch = lora_post_dispatch
# else:
#     param_data = {}
#     post_dispatch = lambda x: x
#
# llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch, attn="eager")
# load_weights(llm, param_data, param_name=None, no_vision=True)
# llm.eval()
#
# _, _, test_ds = get_ds("hoi")
# all_ds = [y for x in range(len(test_ds)) for y in test_ds[x][1]]
# vectors = get_cat2_dataset_means(llm, processor, all_ds, HOI_DATASET_PATH)
# np.save(f"vectors_cache_sim_{sim}.npy", vectors)

sim = False
# sim = True
vectors = np.load(f"vectors_cache_sim_{sim}.npy")

import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
import numpy as np

neighbors_list = [5, 10, 15, 20]
fig, axes = plt.subplots(1, 4, figsize=(6, 1.5))
print(f"Running Isomap sweep on {len(vectors)} vectors...")

for i, k in enumerate(neighbors_list):
    ax = axes[i]
    print(f"  > Computing Isomap (k={k})...")
    iso = Isomap(n_neighbors=k, n_components=2, n_jobs=-1)
    vectors_iso = iso.fit_transform(vectors)
    center = np.mean(vectors_iso, axis=0)
    distances = np.linalg.norm(vectors_iso - center, axis=1)

    ax.scatter(
        vectors_iso[:, 0],
        vectors_iso[:, 1],
        c=distances,
        cmap='plasma_r',
        alpha=0.8,
        s=2,
        edgecolors='none'
    )
    ax.set_title(f"k={k}", fontsize=8, pad=2)
    ax.axis('off')

plt.tight_layout(pad=0.5)
filename = f"isomap_sim_{sim}.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Saved mini sweep to: {filename}")
plt.show()