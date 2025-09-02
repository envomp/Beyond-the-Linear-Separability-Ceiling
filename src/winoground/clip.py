import torch
import torch.nn.functional as F

from scripts.conf import *
from datasets import load_dataset
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

model_id = "openai/clip-vit-large-patch14-336"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)

dataset = load_dataset("facebook/winoground", split="test")

text_score_correct = 0
image_score_correct = 0
group_score_correct = 0
total_examples = len(dataset)

for example in tqdm(dataset):
    image_0 = example['image_0'].convert("RGB")
    image_1 = example['image_1'].convert("RGB")
    caption_0 = example['caption_0']
    caption_1 = example['caption_1']

    image_inputs = processor(images=[image_0, image_1], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)

    text_inputs = tokenizer([caption_0, caption_1], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    image_features_norm = F.normalize(image_features, p=2, dim=1)
    text_features_norm = F.normalize(text_features, p=2, dim=1)
    sim_c0_i0 = torch.dot(text_features_norm[0], image_features_norm[0])
    sim_c0_i1 = torch.dot(text_features_norm[0], image_features_norm[1])
    sim_c1_i0 = torch.dot(text_features_norm[1], image_features_norm[0])
    sim_c1_i1 = torch.dot(text_features_norm[1], image_features_norm[1])

    # Text Score (Image-to-Text): Both images must correctly retrieve their corresponding captions.
    # Paper Eq. 1: s(C0, I0) > s(C1, I0) AND s(C1, I1) > s(C0, I1)
    text_correct_0 = sim_c0_i0 > sim_c1_i0
    text_correct_1 = sim_c1_i1 > sim_c0_i1
    is_text_match = text_correct_0 and text_correct_1

    # Image Score (Text-to-Image): Both captions must correctly retrieve their corresponding images.
    # Paper Eq. 2: s(C0, I0) > s(C0, I1) AND s(C1, I1) > s(C1, I0)
    image_correct_0 = sim_c0_i0 > sim_c0_i1
    image_correct_1 = sim_c1_i1 > sim_c1_i0
    is_image_match = image_correct_0 and image_correct_1

    if is_text_match:
        text_score_correct += 1
    if is_image_match:
        image_score_correct += 1
    if is_text_match and is_image_match:
        group_score_correct += 1

text_accuracy = (text_score_correct / total_examples) * 100 if total_examples > 0 else 0
image_accuracy = (image_score_correct / total_examples) * 100 if total_examples > 0 else 0
group_accuracy = (group_score_correct / total_examples) * 100 if total_examples > 0 else 0

print("\n--- Evaluation Results (Winoground Metrics) ---")
print(f"Total examples: {total_examples}\n")
print(f"Text retrieval score accuracy: {text_accuracy:.2f}% ({text_score_correct}/{total_examples})")
print(f"Image retrieval score accuracy: {image_accuracy:.2f}% ({image_score_correct}/{total_examples})")
print(f"Group score accuracy: {group_accuracy:.2f}% ({group_score_correct}/{total_examples})")

# Text retrieval score accuracy: 27.75% (111/400)
# Image retrieval score accuracy: 11.75% (47/400)
# Group score accuracy: 7.75% (31/400)
