import torch
from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, raw_forward, inference
from src.processor import *
import json
import torch.nn as nn
import torch.optim as optim
import random

trainable_image_size = 336
phi, processor = load_phi_3_5_vision(trainable_image_size=trainable_image_size, do_checkpoint=True)
cat_text, cat_labels, longest_label = get_cat_text_label(processor, "</s>")

with open("../bongard_openworld.json", 'r') as f:
    data = json.load(f)

train, validate, test = data[501:1001], data[1001:1101], data[0:500]

print(f"len train={len(train)}, len validate={len(validate)}")
overlap = lambda a, b: set([x["uid"] for x in a]) & set([x["uid"] for x in b])
print(f"overlap of train and validate: {overlap(train, validate)}")
print(f"overlap of train and test: {overlap(train, test)}")
print(f"overlap of validate and test: {overlap(test, validate)}")

num_epochs = 10
batch_size = 2

optimizer = optim.AdamW([phi.model.vision_embed_tokens.img_processor.trainable_embeddings], lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    phi.train()

    total_loss = 0
    random.shuffle(train)
    for i in range(0, len(train), batch_size):
        batch = train[i:i + batch_size]
        optimizer.zero_grad()

        batch_loss = 0
        for item in batch:
            urls, test_cat = get_urls_and_cat_from_item(item)
            expected_text = cat_text[test_cat]
            labels = cat_labels[test_cat]
            prompt = get_phi_simple_prompt(image_count=len(urls)) + expected_text
            logits = raw_forward(model=phi, processor=processor, urls=urls, prompt=prompt, resize=224)
            answer_logits = logits[:, -longest_label - 1:-1, :]
            print("got:", processor.batch_decode(torch.argmax(answer_logits, dim=-1), skip_special_tokens=True), "| expected:", processor.batch_decode(labels, skip_special_tokens=True))

            loss = loss_fn(answer_logits.transpose(1, 2), labels) / batch_size
            loss.backward()

            batch_loss += loss.item()

        optimizer.step()

        total_loss += batch_loss
        print(f"epoch: {epoch}, step: {i}, loss: {round(batch_loss / batch_size, 6)}")

    correct = 0
    phi.eval()
    for i in range(0, len(validate)):
        urls, test_cat = get_urls_and_cat_from_item(validate[i])
        expected_text = cat_text[test_cat]
        prompt = get_phi_simple_prompt(image_count=len(urls))
        decoded = inference(model=phi, processor=processor, urls=urls, prompt=prompt, max_tokens=100, force_cuda=True)
        if decoded[0].startswith(expected_text):
            correct += 1

    for name, param in phi.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, data: {param.data.flatten()[:5]} ...")
            filename = f"{name}_e_{epoch}_ti_{trainable_image_size}_acc_{correct}.pt"
            torch.save(param.data, filename)
            print(f"Saved trainable parameter '{name}' to '{filename}' with accuracy: {correct} and loss: {round(total_loss / len(train), 6)}")
