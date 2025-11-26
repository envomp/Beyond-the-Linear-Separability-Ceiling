from scripts.conf import *
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from baseline import load_hoi_prototype_data, load_aokvqa_data, load_scienceqa_data, load_pope_data
from PIL import Image
import torch
import torch.nn.functional as F

model_id = "openai/clip-vit-large-patch14-336"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)

DATASET_TO_RUN = "pope"  # aokvqa or gqa or hoi or scienceqa or pope

if DATASET_TO_RUN == "aokvqa":
    vqa_dataset = load_aokvqa_data(split="validation")
elif DATASET_TO_RUN == "hoi":
    vqa_dataset = load_hoi_prototype_data("bongard_hoi_vqa.json", HOI_DATASET_PATH, split_prefix="test")[0]
elif DATASET_TO_RUN == "scienceqa":
    vqa_dataset = load_scienceqa_data(split="test")
elif DATASET_TO_RUN == "pope":
    vqa_dataset = load_pope_data(split="test")
else:
    raise ValueError(f"Unknown DATASET_TO_RUN: {DATASET_TO_RUN}")

correct_predictions = 0
total_samples = len(vqa_dataset)

for item in vqa_dataset:
    if item["image_path"]:
        image = Image.open(item["image_path"]).convert("RGB")
    else:
        image = item["image_object"]
    image_inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_features_norm = F.normalize(image_features, p=2, dim=1)

    text_prompts = []
    choice_letters = []
    for letter, choice_text in item['choices_with_letters'].items():
        prompt = f"Question: {item['question']} Answer: {choice_text}"
        text_prompts.append(prompt)
        choice_letters.append(letter)

    text_inputs = tokenizer(text_prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_features_norm = F.normalize(text_features, p=2, dim=1)
    logits_per_image = image_features_norm @ text_features_norm.T
    prediction_idx = logits_per_image.argmax().item()
    predicted_letter = choice_letters[prediction_idx]
    predicted_letter = "A"

    if predicted_letter == item['correct_letter']:
        correct_predictions += 1

lsc_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

print("\n--- VQA LSC evaluation results ---")
print(f"Dataset: {DATASET_TO_RUN}")
print(f"Model: {model_id}\n")
print(f"VQA LSC accuracy: {lsc_accuracy:.2f}% ({correct_predictions}/{total_samples})")

# we converted hoi and pope tasks from binary classification task to multiple choice tasks operating on a single image,
# where only 1 answer is correct and 3 other are wrong

# HOI:         78.75% (6300/8000)    first: 25.04% (2003/8000)    baseline: 78.89%
# POPE:        62.53% (938/1500)     first: 25.00% (375/1500)     baseline: 79.65%
# GQA:         64.80% (7307/11276)   first: 30.34% (3421/11276)   baseline: 77.76%
# A-OKVQA:     61.14% (700/1145)     first: 24.54% (281/1145)     baseline: 74.34%
# ScienceQA:   46.21% (932/2017)     first: 35.35% (713/2017)     baseline: 87.42%
