from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, load_pixtral_12B, load_gemma3_4B, load_gemma3_27B
from scripts.hf_models import inference, lora_post_dispatch, resize_images
from scripts.hf_models import find_image_token_ranges_phi, find_image_token_ranges_pixtral, find_image_token_ranges_gemma
from src.processor import *
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import math

seed = random.randint(1, 10000)
print("Fixing seed to: ", seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataset = "none"  # hoi or openworld
result_dir = BONGARD_RESULT_DIR
model = "none"  # phi or pixtral or gemma3_4b or gemma3_27b
learnable_embedding_location = "none"  # full, lora
trainable_prompt_size = 0  # 100 for postfix and full, 0 for lora
contrastive_schedule = "constant"  # constant, linear, sine, cosinel
cl_scaling = 0.0
num_epochs = 0
batch_size = 25
eps = 1e-8
contrastive_temperature = 0.07
learning_rate = 0.0001
smooth_cl = False
train_limit = 500
val_limit = 25
resolution = 224

exec(open('configurator.py').read())


if dataset == "hoi":
    dataset_path = HOI_DATASET_PATH
elif dataset == "openworld":
    dataset_path = OPENWORLD_DATASET_PATH
else:
    raise ValueError("Unrecognized dataset")

print(f"params: {dataset} {model} {learnable_embedding_location}")

tsize = trainable_prompt_size * 2 if learnable_embedding_location == "full" else trainable_prompt_size
post_dispatch = lora_post_dispatch if learnable_embedding_location == "lora" else lambda x: x

if model == "phi":
    num_crops = 1 if learnable_embedding_location == "lora" else 4
    llm, processor = load_phi_3_5_vision(trainable_prompt_size=tsize, do_checkpoint=True, num_crops=num_crops, post_dispatch=post_dispatch)
    if learnable_embedding_location == "full":
        trainable_params = [llm.model.vision_embed_tokens.trainable_embeddings]
        trainable_prompt = lambda x: get_phi_trainable_prompt_full(x, trainable_prompt_size)
    elif learnable_embedding_location == "postfix":
        trainable_params = [llm.model.vision_embed_tokens.trainable_embeddings]
        trainable_prompt = lambda x: get_phi_trainable_prompt(x, trainable_prompt_size)
    elif learnable_embedding_location == "lora":
        trainable_params = [p for n, p in llm.named_parameters() if p.requires_grad]
        trainable_prompt = lambda x: get_phi_simple_prompt(x)
    else:
        raise ValueError("Unknown learnable embedding location: " + learnable_embedding_location)
    ranges = find_image_token_ranges_phi
elif model == "pixtral":
    llm, processor = load_pixtral_12B(trainable_prompt_size=tsize, do_checkpoint=True, post_dispatch=post_dispatch)
    if learnable_embedding_location == "full":
        trainable_params = [llm.language_model.get_input_embeddings().trainable_embeddings]
        trainable_prompt = lambda x: get_pixtral_trainable_prompt_full(x, trainable_prompt_size)
    elif learnable_embedding_location == "postfix":
        trainable_params = [llm.language_model.get_input_embeddings().trainable_embeddings]
        trainable_prompt = lambda x: get_pixtral_trainable_prompt(x, trainable_prompt_size)
    elif learnable_embedding_location == "lora":
        trainable_params = [p for n, p in llm.named_parameters() if p.requires_grad]
        trainable_prompt = lambda x: get_pixtral_simple_prompt(x)
    else:
        raise ValueError("Unknown learnable embedding location: " + learnable_embedding_location)
    ranges = find_image_token_ranges_pixtral
elif model == "gemma3_4b":
    llm, processor = load_gemma3_4B(trainable_prompt_size=tsize, do_checkpoint=True, post_dispatch=post_dispatch)
    if learnable_embedding_location == "full":
        trainable_params = [llm.language_model.get_input_embeddings().trainable_embeddings]
        trainable_prompt = lambda x: get_gemma_trainable_prompt_full(x, trainable_prompt_size)
    elif learnable_embedding_location == "postfix":
        trainable_params = [llm.language_model.get_input_embeddings().trainable_embeddings]
        trainable_prompt = lambda x: get_gemma_trainable_prompt(x, trainable_prompt_size)
    elif learnable_embedding_location == "lora":
        trainable_params = [p for n, p in llm.named_parameters() if p.requires_grad]
        trainable_prompt = lambda x: get_gemma_simple_prompt(x)
    else:
        raise ValueError("Unknown learnable embedding location: " + learnable_embedding_location)
    ranges = find_image_token_ranges_gemma
else:
    raise ValueError("Invalid model name: " + model)

cat_text, cat_labels, longest_label = get_cat_text_label(processor, processor.tokenizer.eos_token)

train, validate, test = get_ds(dataset, train_limit=train_limit, val_limit=val_limit)

optimizer = optim.AdamW(trainable_params, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


def get_cosinel(cur, of_total):
    wc = math.cos(math.pi / 2 * ((cur + 1) / of_total))
    wn = 1 - wc
    w2 = min((2 * (cur + 1) / of_total), 1)
    wc *= w2 * cl_scaling
    return wc, wn


for epoch in range(num_epochs):
    if contrastive_schedule == "linear":
        wc = ((epoch + 1) / num_epochs) * cl_scaling
        wn = 1
    elif contrastive_schedule == "sine":
        wc = math.sin(math.pi / 2 * ((epoch + 1) / num_epochs)) * cl_scaling
        wn = 1
    elif contrastive_schedule == "cosinel":
        wc, wn = get_cosinel(epoch, num_epochs)
        print(f"wc={wc}, wn={wn}")
    elif contrastive_schedule == "constant":
        wc = cl_scaling
        wn = 1
    else:
        raise ValueError("Unknown contrastive schedule: " + contrastive_schedule)
    llm.train()

    total_loss = 0
    random.shuffle(train)
    for e in range(0, len(train), batch_size):
        batch = train[e:e + batch_size]
        optimizer.zero_grad()

        if smooth_cl and contrastive_schedule == "cosinel":
            wc, wn = get_cosinel(epoch * len(train) + e, len(train) * num_epochs)

        batch_loss = 0
        for item in batch:
            urls, test_cat = get_urls_and_cat_from_item(item, dataset_path)
            expected_text = cat_text[test_cat]
            labels = cat_labels[test_cat]
            prompt = trainable_prompt(len(urls)) + expected_text
            images = [Image.open(url).convert("RGB") for url in urls]
            if resolution:
                images = resize_images(images, longest_edge=resolution)
            inputs = processor(images=images, text=prompt, return_tensors="pt").to(llm.dtype).to("cuda")
            out = llm(**inputs, output_hidden_states=True)
            indexes = ranges(inputs)
            answer_logits = out["logits"][:, -longest_label - 1:-1, :]
            print(processor.batch_decode(torch.argmax(answer_logits, dim=-1), skip_special_tokens=False))

            last_layer_hidden_states = out["hidden_states"][-1]
            test_img = last_layer_hidden_states[0][indexes[0][len(urls)][0]:indexes[0][len(urls)][1]]
            cat_imgs = len(urls) // 2
            cat2_imgs, cat1_imgs = [], []
            for i in range(1, cat_imgs + 1):
                cat2_imgs.append(last_layer_hidden_states[0][indexes[0][i][0]:indexes[0][i][1], :])
            for i in range(cat_imgs + 1, 2 * cat_imgs + 1):
                cat1_imgs.append(last_layer_hidden_states[0][indexes[0][i][0]:indexes[0][i][1], :])
            maximize_imgs, minimize_imgs = (cat2_imgs, cat1_imgs) if test_cat == "cat_2" else (cat1_imgs, cat2_imgs)

            max_cat, min_cat = torch.cat(maximize_imgs, dim=0), torch.cat(minimize_imgs, dim=0)
            test_pool, max_pool, min_pool = torch.mean(test_img, dim=0), torch.mean(max_cat, dim=0), torch.mean(min_cat, dim=0)
            test_norm, max_norm, min_norm = F.normalize(test_pool, p=2, dim=0, eps=eps), F.normalize(max_pool, p=2, dim=0, eps=eps), F.normalize(min_pool, p=2, dim=0, eps=eps)
            sim_positive, sim_negative = torch.dot(test_norm, max_norm), torch.dot(test_norm, min_norm)
            contrastive_logits = torch.stack([sim_positive, sim_negative], dim=0) / contrastive_temperature
            contrastive_target = torch.tensor([0], device=test_pool.device)

            next_token_loss = wn * loss_fn(answer_logits.transpose(1, 2), labels)
            contrastive_loss = wc * loss_fn(contrastive_logits.unsqueeze(0), contrastive_target)
            loss = next_token_loss + contrastive_loss
            loss /= batch_size
            loss.backward()  # gradient accumulation
            batch_loss += loss.item()

        optimizer.step()

        total_loss += batch_loss
        print(f"epoch: {epoch}, step: {e}, loss: {round(batch_loss, 6)}, wc: {wc}, wn: {wn}")

    correct = [0] * len(validate)
    llm.eval()
    for j, (name, val_split) in enumerate(validate):
        for i in range(0, len(val_split)):
            urls, test_cat = get_urls_and_cat_from_item(val_split[i], dataset_path)
            prompt = trainable_prompt(len(urls)) + "answer:"
            decoded = inference(model=llm, processor=processor, urls=urls, prompt=prompt, max_tokens=100, resize=resolution, force_cuda=True)
            got = decoded[0].strip()
            print("got:", got, "| expected:", test_cat, "| equals:", got.startswith(test_cat))
            if got.startswith(test_cat):
                correct[j] += 1

    state_tensors = {}
    for name, param in llm.named_parameters():
        if param.requires_grad:
            state_tensors[name] = param
            print(f"> Will save {name}, dtype={param.dtype}, shape:{param.data.shape} ...")
    filename = result_dir + f"sim_{dataset}_{model}_{learnable_embedding_location}_c_{cl_scaling}_e_{epoch}_t_{trainable_prompt_size}_acc_{"_".join([str(x) for x in correct])}_seed_{seed}.pt"
    torch.save(state_tensors, filename)
    print(f"Saving state_dict to '{filename}' with accuracy: {correct} and loss: {round(total_loss / (len(train) / batch_size), 6)}")
