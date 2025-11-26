from scripts.conf import *
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import GenerationConfig
from matplotlib.lines import Line2D

from scripts.hf_models import load_phi_3_5_vision, load_weights, lora_post_dispatch, resize_images
from scripts.hf_models import find_image_token_ranges_phi
from processor import get_ds, get_urls_and_cat_from_item, get_phi_simple_prompt, interleaved_prompt

param_datas = {
    ("phi", None, None, None): None,
    ("phi", "lora", "hoi", False): "sim_hoi_phi_lora_c_0.0_e_2_t_0_acc_74_81_77_82_seed_9712.pt",
    ("phi", "lora", "hoi", True): "sim_hoi_phi_lora_c_0.4_e_2_t_0_acc_79_80_81_82_seed_4645.pt",
}


def visualize_manifold_with_tokens(model_name, learnable_embedding_location, dataset, sim, layer):
    """
    Visualizes the embedding manifold for a single Bongard problem using PCA.
    Plots all individual tokens per image and their mean-pooled vector.
    Uses color gradients to differentiate images within a category.
    """
    print(f"Loading model: {model_name}, Location: {learnable_embedding_location}, Dataset: {dataset}, Sim: {sim}, Layer: {layer}")
    dataset_path = HOI_DATASET_PATH
    param_key = (model_name, learnable_embedding_location, dataset, sim)
    param_file = param_datas.get(param_key)
    if not param_file:
        param_data = {}
    else:
        param_file = PEFT_PATH + param_file
        param_data = torch.load(param_file, weights_only=True)
    post_dispatch = lora_post_dispatch if param_file else lambda x: x
    ranges = find_image_token_ranges_phi
    prompt_fn = lambda x: get_phi_simple_prompt(x, base_prompt=interleaved_prompt)
    llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch)
    load_weights(llm, param_data, param_name=None, no_vision=True)
    llm.eval()

    # --- 2. Data Loading ---
    train, _, _ = get_ds("hoi")
    item = train[8] # using the images we used in article
    print(item)
    urls, test_cat = get_urls_and_cat_from_item(item, dataset_path)
    num_images_per_cat = 6
    images = resize_images([Image.open(url).convert("RGB") for url in urls], longest_edge=224)

    # --- 3. Inference & Embedding Extraction ---
    inputs = processor(images=images, text=prompt_fn(len(images)), return_tensors="pt").to(llm.dtype).to("cuda")
    cfg = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True, use_cache=False)
    with torch.no_grad():
        outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id, generation_config=cfg)

    layer_hidden_states = outputs["hidden_states"][0][layer][0] # Shape: [sequence_length, hidden_dim]
    indexes = ranges(inputs)

    def aggregate_tokens(tokens_tensor):
        """Pools a tensor of tokens into a single vector."""
        return tokens_tensor.mean(dim=0)

    # --- 4. Get All Token and Mean Embeddings ---

    test_tokens = layer_hidden_states[indexes[0][len(urls)][0]:indexes[0][len(urls)][1]].detach().float().cpu()
    pos_tokens_list = []
    neg_tokens_list = []
    for i in range(1, num_images_per_cat + 1):
        pos_tokens_list.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :].detach().float().cpu())
    for i in range(num_images_per_cat + 1, 2 * num_images_per_cat + 1):
        neg_tokens_list.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :].detach().float().cpu())
    all_image_tokens = [test_tokens] + pos_tokens_list + neg_tokens_list
    mean_arrays_1d = [aggregate_tokens(t).numpy() for t in all_image_tokens]
    token_arrays_2d = [t.numpy() for t in all_image_tokens]
    mean_arrays_2d = [m.reshape(1, -1) for m in mean_arrays_1d]
    all_vectors_list = token_arrays_2d + mean_arrays_2d

    token_labels = []
    token_image_group = []
    token_labels.extend(['test_token'] * len(test_tokens))
    token_image_group.extend([0] * len(test_tokens))
    for i, tokens in enumerate(pos_tokens_list, 1):
        token_labels.extend(['pos_token'] * len(tokens))
        token_image_group.extend([i] * len(tokens))
    for i, tokens in enumerate(neg_tokens_list, 7):
        token_labels.extend(['neg_token'] * len(tokens))
        token_image_group.extend([i] * len(tokens))

    mean_labels = ['test_mean',
                   'pos_mean', 'pos_mean', 'pos_mean', 'pos_mean', 'pos_mean', 'pos_mean',
                   'neg_mean', 'neg_mean', 'neg_mean', 'neg_mean', 'neg_mean', 'neg_mean']
    mean_image_group = list(range(13))
    all_labels = np.array(token_labels + mean_labels)
    all_image_groups = np.array(token_image_group + mean_image_group)
    all_vectors_high_dim = np.concatenate(all_vectors_list, axis=0)
    pos_means_high_dim = np.array(mean_arrays_1d[1:7])
    neg_means_high_dim = np.array(mean_arrays_1d[7:13])
    pos_class_centroid = pos_means_high_dim.mean(axis=0).reshape(1, -1)
    neg_class_centroid = neg_means_high_dim.mean(axis=0).reshape(1, -1)

    # --- 5. PCA Dimensionality Reduction ---
    print("Running PCA on all tokens and mean-pooled embeddings...")
    pca = PCA(n_components=2, random_state=42)
    all_vectors_2d = pca.fit_transform(all_vectors_high_dim)
    pos_class_centroid_2d = pca.transform(pos_class_centroid)
    neg_class_centroid_2d = pca.transform(neg_class_centroid)

    # --- 6. Plotting ---
    print("Generating plot...")
    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    pos_colors = [blues(i) for i in np.linspace(0.3, 0.9, num_images_per_cat)]
    neg_colors = [reds(i) for i in np.linspace(0.3, 0.9, num_images_per_cat)]
    test_color_token = 'palegreen'
    test_color_mean = 'green'

    image_group_colors_tokens = {0: test_color_token}
    image_group_colors_tokens.update(zip(range(1, 7), pos_colors))
    image_group_colors_tokens.update(zip(range(7, 13), neg_colors))

    image_group_colors_means = {0: test_color_mean}
    image_group_colors_means.update(zip(range(1, 7), pos_colors))
    image_group_colors_means.update(zip(range(7, 13), neg_colors))

    plt.figure(figsize=(5, 3))

    for i in range(13): # 0=test, 1-6=pos, 7-12=neg
        mask = (all_labels == 'test_token') | (all_labels == 'pos_token') | (all_labels == 'neg_token')
        mask_group = (all_image_groups == i) & mask

        plt.scatter(
            all_vectors_2d[mask_group, 0],
            all_vectors_2d[mask_group, 1],
            color=image_group_colors_tokens[i],
            s=15,
            alpha=0.2,
            label=f'Image {i} tokens' if i==0 else None
        )

    plt.scatter(
        pos_class_centroid_2d[:, 0],
        pos_class_centroid_2d[:, 1],
        c='blue',
        label='cat_2 Class Centroid (LSC)',
        s=500,
        marker='X',
        edgecolors='black',
        linewidth=2
    )
    plt.scatter(
        neg_class_centroid_2d[:, 0],
        neg_class_centroid_2d[:, 1],
        c='red',
        label='cat_1 Class Centroid (LSC)',
        s=500,
        marker='X',
        edgecolors='black',
        linewidth=2
    )

    for i in range(13):
        mask = (all_labels == 'test_mean') # | (all_labels == 'pos_mean') | (all_labels == 'neg_mean')
        mask_group = (all_image_groups == i) & mask

        plt.scatter(
            all_vectors_2d[mask_group, 0],
            all_vectors_2d[mask_group, 1],
            c=image_group_colors_means[i],
            s=250,
            marker='P',
            edgecolors='black',
            linewidth=1
        )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='query image tokens', markerfacecolor=test_color_token, markersize=10, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='positive image tokens', markerfacecolor='lightblue', markersize=10, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='negative image tokens', markerfacecolor='lightcoral', markersize=10, alpha=0.5),
        Line2D([0], [0], marker='P', color='w', label='query image (mean)', markerfacecolor=test_color_mean, markeredgecolor='black', markersize=12),
        # Line2D([0], [0], marker='P', color='w', label='cat_2 images (mean)', markerfacecolor='cornflowerblue', markeredgecolor='black', markersize=12),
        # Line2D([0], [0], marker='P', color='w', label='cat_1 images (mean)', markerfacecolor='salmon', markeredgecolor='black', markersize=12),
        Line2D([0], [0], marker='X', color='w', label='positive set centroid', markerfacecolor='blue', markeredgecolor='black', markersize=15, mew=2),
        Line2D([0], [0], marker='X', color='w', label='negative set centroid', markerfacecolor='red', markeredgecolor='black', markersize=15, mew=2),
    ]

    plt.xlabel(f"Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=10)
    plt.ylabel(f"Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=10)
    plt.legend(handles=legend_elements, fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_filename = f"manifold_pca_layer_{layer}_sim_{sim}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")



# 1. Visualize the L_combined model (sim=True)
visualize_manifold_with_tokens(
    model_name="phi",
    learnable_embedding_location="lora",
    dataset="hoi",
    sim=True,
    layer=-1  # Use -1 for the final hidden state
)

# 2. Visualize the L_NT-only model (sim=False)
visualize_manifold_with_tokens(
    model_name="phi",
    learnable_embedding_location="lora",
    dataset="hoi",
    sim=False,
    layer=-1
)

# 3. Visualize the baseline model (no PEFT)
visualize_manifold_with_tokens(
    model_name="phi",
    learnable_embedding_location=None,
    dataset=None,
    sim=None,
    layer=-1
)
