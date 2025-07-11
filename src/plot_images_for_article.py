import json

from scripts.conf import *
import matplotlib.pyplot as plt
from PIL import Image


def plot_image(full_path, ax):
    try:
        image = Image.open(full_path).convert("RGB")
        ax.imshow(image)
        ax.axis('off')
    except FileNotFoundError:
        print(f"Error: Image not found at {full_path}")
        ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=9, color='red')
        ax.set_facecolor('#f0f0f0')
        ax.tick_params(axis='both', which='both', length=0)

def save_image_set(image_paths, title, save_path):
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2.5, 2.5))
    fig.suptitle(title, fontsize=16, weight='bold')

    if num_images == 1:
        axes = [axes]

    for i, rel_path in enumerate(image_paths):
        full_path = os.path.join(HOI_DATASET_PATH, rel_path)
        plot_image(full_path, axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    print(f"Saving figure to {save_path}...")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Done.")
    plt.close(fig)

def save_single_image(image_path, title, save_path):
    full_path = os.path.join(HOI_DATASET_PATH, image_path)

    try:
        with Image.open(full_path) as img:
            width, height = img.size
            aspect_ratio = height / width if width > 0 else 1
    except FileNotFoundError:
        aspect_ratio = 1.0 # Default to a square if image not found

    base_width = 3
    fig, ax = plt.subplots(figsize=(base_width, base_width * aspect_ratio))
    ax.set_title(title, fontsize=14, weight='bold')
    plot_image(full_path, ax)

    plt.tight_layout()

    print(f"Saving figure to {save_path}...")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    print("Done.")
    plt.close(fig)

if __name__ == '__main__':
    with open("../bongard_hoi.json", 'r') as f:
        data = json.load(f)

    samples = data["train"][8]
    print("Processing sample:", samples)

    positive_example_urls = samples["imagefiles"]["cat_2"][:3]
    negative_example_urls = samples["imagefiles"]["cat_1"][:3]
    test_image_url = samples["testfiles"]["testimage"]

    save_image_set(positive_example_urls, "Positive Examples", "positive_examples.png")
    save_image_set(negative_example_urls, "Negative Examples", "negative_examples.png")
    save_single_image(test_image_url, "Query Image to Classify", "query_image.png")
