from scripts.conf import *
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from baseline import load_gqa_data, load_aokvqa_data
from PIL import Image
import torch
import torch.nn.functional as F

model_id = "openai/clip-vit-large-patch14-336"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)

DATASET_TO_RUN = "aokvqa"  # aokvqa or gqa

if DATASET_TO_RUN == "gqa":
    vqa_dataset = load_gqa_data("gqa_contrastive_pairs_eval.json", GQA_IMAGE_DIR)
    dataset_name_str = "gqa"
elif DATASET_TO_RUN == "aokvqa":
    vqa_dataset = load_aokvqa_data(split="validation")
    dataset_name_str = "aokvqa"
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

    if predicted_letter == item['correct_letter']:
        correct_predictions += 1

lsc_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

print("\n--- VQA LSC Evaluation Results ---")
print(f"Dataset: {dataset_name_str}")
print(f"Model: {model_id}\n")
print(f"VQA LSC Accuracy: {lsc_accuracy:.2f}% ({correct_predictions}/{total_samples})")


# GQA         64.80% (7307/11276)
# A-OKVQA     61.14% (700/1145)
