from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, load_pixtral_12B, load_gemma3_4B
from scripts.hf_models import inference, load_weights, lora_post_dispatch
from processor import *
from best_PEFT import param_datas


def eval_generative(model, learnable_embedding_location, dataset, sim, base_prompt=interleaved_prompt, cross_domain=False, no_vision=False):
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
        if learnable_embedding_location == "full":
            prompt_fn = lambda x: get_gemma_trainable_prompt_full(x, trainable_tokens=trainable_prompt_size, base_prompt=base_prompt)
        elif learnable_embedding_location == "postfix":
            prompt_fn = lambda x: get_gemma_trainable_prompt(x, trainable_prompt_size)
        elif learnable_embedding_location == "lora":
            prompt_fn = lambda x: get_gemma_simple_prompt(x, base_prompt=base_prompt)
    else:
        raise ValueError(f"Unknown model type: {model}")

    load_weights(llm, param_data, param_name=param_name, no_vision=no_vision)
    cat_text, _, _ = get_cat_text_label(processor)

    _, _, test = get_ds(dataset)

    _sim = '_sim' if sim else ''
    _no_vision = '_no_vision' if no_vision else ''
    _cross_domain = '_cross_domain' if cross_domain else ''
    _res = f"_{resolution}" if resolution != 224 else ""
    _promt = f"_{base_prompt.__name__}" if base_prompt.__name__ != "interleaved_prompt" else ""
    run_filename = f"{model}{_cross_domain}_{dataset}{_sim}_{learnable_embedding_location}{_no_vision}{_res}_eval{_promt}.txt"
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
            for i in range(0, len(test_split)):
                urls, test_cat = get_urls_and_cat_from_item(test_split[i], dataset_path)
                prompt = prompt_fn(len(urls)) + "answer:"
                decoded = inference(model=llm, processor=processor, urls=urls, prompt=prompt, max_tokens=10, resize=resolution, force_cuda=True)
                got = decoded[0].strip().replace("cat1", "cat_1").replace("cat2", "cat_2")
                f_out.write(f"got: {got} | expected: {test_cat} | equals: {got.startswith(test_cat)}\n")
                if got.startswith(test_cat):
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
        run_filename = f"results_generative.txt"
        full_file_path = os.path.join(PEFT_PATH, run_filename)
        with open(full_file_path, 'a', encoding='utf-8', buffering=1) as f_out:
            f_out.write(f"{model} {dataset} {learnable_embedding_location} {sim} {correct}\n")


# structural & domain generalization
for model, learnable_embedding_location, dataset, sim in param_datas:
    eval_generative(model, learnable_embedding_location, dataset, sim, base_prompt=interleaved_prompt)
    eval_generative(model, learnable_embedding_location, dataset, sim, base_prompt=labeled_prompt)
    if learnable_embedding_location in ["lora"]:
        eval_generative(model, learnable_embedding_location, dataset, sim, base_prompt=interleaved_prompt, no_vision=True)
        eval_generative(model, learnable_embedding_location, dataset, sim, base_prompt=labeled_prompt, no_vision=True)
    if learnable_embedding_location in ["postfix", "full", "lora"]:
        eval_generative(model, learnable_embedding_location, dataset, sim, base_prompt=interleaved_prompt, cross_domain=True)
