from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, load_pixtral_12B, load_gemma3_4B
from scripts.hf_models import load_weights, lora_post_dispatch, resize_images
from scripts.hf_models import find_image_token_ranges_phi, find_image_token_ranges_pixtral, find_image_token_ranges_gemma
from transformers import GenerationConfig
from PIL import Image
from scripts.eval_similarity import evaluate_pairs_b
from src.processor import *
from best_PEFT import param_datas


def eval_similarity(model, learnable_embedding_location, dataset, sim, layer, base_prompt=interleaved_prompt, cross_domain=False, no_vision=False):
    if dataset == "hoi":
        dataset_path = HOI_DATASET_PATH
    elif dataset == "openworld":
        dataset_path = OPENWORLD_DATASET_PATH
    else:
        raise ValueError(f"Dataset not supported: {dataset}")

    if not cross_domain:
        load_dataset = dataset
    else:
        if dataset == "openworld":
            load_dataset = "hoi"
        elif dataset == "hoi":
            load_dataset = "openworld"

    param_data = torch.load(PEFT_PATH + param_datas[(model, learnable_embedding_location, load_dataset, sim)], weights_only=True)
    param_name = None

    resolution = 224
    trainable_prompt_size = 100 if learnable_embedding_location in ["full", "postfix"] else 0
    trainable_image_size = 336 if learnable_embedding_location in ["vision"] else 0
    tsize = trainable_prompt_size * 2 if learnable_embedding_location == "full" else trainable_prompt_size
    post_dispatch = lora_post_dispatch if learnable_embedding_location == "lora" else lambda x: x

    if model == "phi":
        llm, processor = load_phi_3_5_vision(trainable_prompt_size=tsize, trainable_image_size=trainable_image_size, post_dispatch=post_dispatch)
        ranges = find_image_token_ranges_phi
        if learnable_embedding_location == "full":
            prompt_fn = lambda x: get_phi_trainable_prompt_full(x, trainable_tokens=trainable_prompt_size, base_prompt=base_prompt)
        elif learnable_embedding_location == "lora":
            prompt_fn = lambda x: get_phi_simple_prompt(x, base_prompt=base_prompt)
        elif learnable_embedding_location == "postfix":
            param_name = "model.vision_embed_tokens.trainable_embeddings"
            prompt_fn = lambda x: get_phi_trainable_prompt(x, trainable_tokens=trainable_prompt_size, base_prompt=base_prompt)
        elif learnable_embedding_location == "vision":
            param_name = "model.vision_embed_tokens.img_processor.trainable_embeddings"
            prompt_fn = lambda x: get_phi_simple_prompt(x, base_prompt=base_prompt)
        elif learnable_embedding_location == "connector":
            prompt_fn = lambda x: get_phi_simple_prompt(x, base_prompt=base_prompt)
    elif model == "pixtral":
        llm, processor = load_pixtral_12B(trainable_prompt_size=tsize, post_dispatch=post_dispatch)
        ranges = find_image_token_ranges_pixtral
        if learnable_embedding_location == "full":
            param_name = "language_model.model.embed_tokens.trainable_embeddings"
            prompt_fn = lambda x: get_pixtral_trainable_prompt_full(x, trainable_tokens=trainable_prompt_size, base_prompt=base_prompt)
        elif learnable_embedding_location == "postfix":
            param_name = "language_model.model.embed_tokens.trainable_embeddings"
            prompt_fn = lambda x: get_pixtral_trainable_prompt(x, trainable_prompt_size)
        elif learnable_embedding_location == "lora":
            prompt_fn = lambda x: get_pixtral_simple_prompt(x, base_prompt=base_prompt)
    elif model == "gemma3_4b":
        llm, processor = load_gemma3_4B(trainable_prompt_size=tsize, post_dispatch=post_dispatch)
        ranges = find_image_token_ranges_gemma
        if learnable_embedding_location == "full":
            prompt_fn = lambda x: get_gemma_trainable_prompt_full(x, trainable_tokens=trainable_prompt_size, base_prompt=base_prompt)
        elif learnable_embedding_location == "postfix":
            prompt_fn = lambda x: get_gemma_trainable_prompt(x, trainable_prompt_size)
        elif learnable_embedding_location == "lora":
            prompt_fn = lambda x: get_gemma_simple_prompt(x, base_prompt=base_prompt)
    else:
        raise ValueError(f"Unknown model type: {model}")

    load_weights(llm, param_data, param_name=param_name, no_vision=no_vision)

    _, _, test = get_ds(dataset)

    _sim = '_sim' if sim else ''
    _no_vision = '_no_vision' if no_vision else ''
    _cross_domain = '_cross_domain' if cross_domain else ''
    _res = f"_{resolution}" if resolution != 224 else ""
    _promt = f"_{base_prompt.__name__}" if base_prompt.__name__ != "interleaved_prompt" else ""
    layer_name = "final" if layer == -1 else "vision" if layer == 0 else str(layer)
    run_filename = f"{model}{_cross_domain}_{dataset}{_sim}_{learnable_embedding_location}{_no_vision}{_res}_similarity_{layer_name}{_promt}.txt"
    full_file_path = os.path.join(PEFT_PATH, run_filename)

    with open(full_file_path, 'w', encoding='utf-8', buffering=1) as f_out:
        f_out.write(f"experiment run details:\n")
        f_out.write(f"  dataset: {dataset}\n")
        f_out.write(f"  model: {model} {load_dataset} {learnable_embedding_location}\n")
        f_out.write(f"  path: {dataset_path}\n")
        f_out.write(f"---------------------------------------\n")
        results = {"correct": {"cat_1": 0, "cat_2": 0}, "incorrect": {"cat_1": 0, "cat_2": 0}}

        correct = [0] * len(test)
        llm.eval()
        for j, (test_split_name, test_split) in enumerate(test):
            f_out.write(f"  test split name: {test_split_name}\n")
            f_out.write(f"---------------------------------------\n\n")

            for item in test_split:
                urls, test_cat = get_urls_and_cat_from_item(item, dataset_path)
                images = resize_images([Image.open(url).convert("RGB") for url in urls], longest_edge=resolution)
                inputs = processor(images=images, text=prompt_fn(len(images)), return_tensors="pt").to(llm.dtype).to("cuda")
                cfg = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True)
                outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id, generation_config=cfg)
                indexes = ranges(inputs)
                layer_hidden_states = outputs["hidden_states"][0][layer][0]
                test_img = layer_hidden_states[indexes[0][len(urls)][0]:indexes[0][len(urls)][1]]
                cat_imgs = len(urls) // 2
                cat2_imgs, cat1_imgs = [], []
                for i in range(1, cat_imgs + 1):
                    cat2_imgs.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :])
                for i in range(cat_imgs + 1, 2 * cat_imgs + 1):
                    cat1_imgs.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :])
                maximize_imgs, minimize_imgs = (cat2_imgs, cat1_imgs) if test_cat == "cat_2" else (cat1_imgs, cat2_imgs)
                max_cat, min_cat = torch.cat(maximize_imgs, dim=0), torch.cat(minimize_imgs, dim=0)
                pairs = [(None, [test_img]), (None, [max_cat]), (None, [min_cat])]
                if evaluate_pairs_b(pairs, True, f_out=f_out):
                    correct[j] += 1
                    results["correct"][test_cat] += 1
                else:
                    results["incorrect"][test_cat] += 1

            split_correct = results["correct"]["cat_1"] + results["correct"]["cat_2"]
            split_incorrect = results["incorrect"]["cat_1"] + results["incorrect"]["cat_2"]
            accuracy = split_correct / max(1, split_correct + split_incorrect) * 100
            f_out.write(f"---------------------------------------\n")
            f_out.write(f"Summary for Split '{test_split_name}':\n")
            f_out.write(f" results: {results}\n")
            f_out.write(f" accuracy: {accuracy:.2f}%\n\n")
            f_out.write(f"---------------------------------------\n")

        print(correct)
        run_filename = f"results_similarity.txt"
        full_file_path = os.path.join(PEFT_PATH, run_filename)
        with open(full_file_path, 'a', encoding='utf-8', buffering=1) as f_out:
            f_out.write(f"{model} {dataset} {learnable_embedding_location} {sim} {layer} {correct}\n")


# structural generalization
for model, learnable_embedding_location, dataset, sim in param_datas:
    for layer in [0, -1]:
        eval_similarity(model, learnable_embedding_location, dataset, sim, layer, base_prompt=interleaved_prompt)
        eval_similarity(model, learnable_embedding_location, dataset, sim, layer, base_prompt=labeled_prompt)
        if learnable_embedding_location in ["postfix", "full", "lora"]:
            eval_similarity(model, learnable_embedding_location, dataset, sim, layer, base_prompt=interleaved_prompt, cross_domain=True)

# no_vision
for layer in [0, -1]:
    eval_similarity("phi", "lora", "openworld", False, layer, base_prompt=interleaved_prompt, no_vision=True)
    eval_similarity("phi", "lora", "openworld", False, layer, base_prompt=labeled_prompt, no_vision=True)
