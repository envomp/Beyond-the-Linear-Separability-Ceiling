import os

import torch
from PIL import Image
from transformers import GenerationConfig

from src.processor import get_ds, get_urls_and_cat_from_item, direct_prompt
from scripts.eval_similarity import evaluate_pairs_b, evaluate_pairs_s
from scripts.hf_models import resize_images


def run_similarity_baselines(llm, processor, model_str, datasets, similarity_layers, prompts_config, ranges_f, get_full_prompt_f, force_cuda=False):
    RESULTS_DIR = f"../../result/{model_str}/similarity_baselines"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    llm.eval()
    for dataset, dataset_path in datasets:
        _, _, test_splits = get_ds(dataset)

        for layer_name, layer in similarity_layers:
            for prompt_structure, test_first, prompt_structure_name in prompts_config:
                run_filename = f"results_{dataset}_{layer_name}_{prompt_structure_name}.txt"
                full_file_path = os.path.join(RESULTS_DIR, run_filename)

                with open(full_file_path, 'w', encoding='utf-8', buffering=1) as f_out:
                    f_out.write(f"experiment run details:\n")
                    f_out.write(f"  dataset: {dataset}\n")
                    f_out.write(f"  path: {dataset_path}\n")
                    f_out.write(f"  similarity location: {layer_name}\n")
                    f_out.write(f"  prompt structure: {prompt_structure_name}\n\n")
                    f_out.write(f"---------------------------------------\n")

                    for test_split_name, test_split in test_splits:
                        f_out.write(f"  test split name: {test_split_name}\n")
                        f_out.write(f"---------------------------------------\n\n")
                        results = {"correct": {"cat_1": 0, "cat_2": 0}, "incorrect": {"cat_1": 0, "cat_2": 0}}
                        for i, item in enumerate(test_split):
                            image_urls, test_cat = get_urls_and_cat_from_item(item, dataset_path)
                            image_count = len(image_urls)
                            cat_imgs = (image_count - 1) // 2
                            if test_first:
                                image_urls = [image_urls[-1]] + image_urls[:-1]
                            images = resize_images([Image.open(url).convert("RGB") for url in image_urls], longest_edge=224)
                            full_prompt = get_full_prompt_f(cat_imgs, image_count, direct_prompt, prompt_structure, test_first, False)
                            inputs = processor(images=images, text=full_prompt, return_tensors="pt").to(llm.dtype)
                            if force_cuda:
                                inputs = inputs.to("cuda")
                            cfg = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True)
                            outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id,
                                                   return_dict_in_generate=True, output_hidden_states=True, generation_config=cfg)
                            out, indexes = outputs["hidden_states"], ranges_f(inputs)
                            layer_hidden_states = out[0][layer][0]
                            cat_imgs = len(image_urls) // 2
                            cat2_imgs, cat1_imgs = [], []
                            if test_first:
                                test_img = layer_hidden_states[indexes[0][1][0]:indexes[0][1][1]]
                                for i in range(2, cat_imgs + 2):
                                    cat2_imgs.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :])
                                for i in range(cat_imgs + 2, 2 * cat_imgs + 2):
                                    cat1_imgs.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :])
                            else:
                                test_img = layer_hidden_states[indexes[0][len(image_urls)][0]:indexes[0][len(image_urls)][1]]
                                for i in range(1, cat_imgs + 1):
                                    cat2_imgs.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :])
                                for i in range(cat_imgs + 1, 2 * cat_imgs + 1):
                                    cat1_imgs.append(layer_hidden_states[indexes[0][i][0]:indexes[0][i][1], :])
                            maximize_imgs, minimize_imgs = (cat2_imgs, cat1_imgs) if test_cat == "cat_2" else (cat1_imgs, cat2_imgs)
                            max_cat, min_cat = torch.cat(maximize_imgs, dim=0), torch.cat(minimize_imgs, dim=0)
                            pairs = [(None, [test_img]), (None, [max_cat]), (None, [min_cat])]

                            if evaluate_pairs_b(pairs, True, f_out=f_out):
                                results["correct"][test_cat] += 1
                            else:
                                results["incorrect"][test_cat] += 1

                        correct = results["correct"]["cat_1"] + results["correct"]["cat_2"]
                        incorrect = results["incorrect"]["cat_1"] + results["incorrect"]["cat_2"]
                        accuracy = correct / max(1, correct + incorrect) * 100
                        f_out.write(f"---------------------------------------\n")
                        f_out.write(f"Summary for Split '{test_split_name}':\n")
                        f_out.write(f" results: {results}\n")
                        f_out.write(f" accuracy: {accuracy:.2f}%\n\n")
                        f_out.write(f"---------------------------------------\n")


def run_similarity_baseline_isolation(llm, processor, model_str, datasets, similarity_layers, prompts_config, force_cuda=False):
    RESULTS_DIR = f"../../result/{model_str}/similarity_baselines"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    llm.eval()

    for dataset, dataset_path in datasets:
        _, _, test_splits = get_ds(dataset)

        for layer_name, layer in similarity_layers:
            for skip_tokens_start, skip_tokens_end, batched, prompt, prompt_structure_name in prompts_config:
                run_filename = f"results_{dataset}_isolation_{layer_name}_{prompt_structure_name}.txt"
                full_file_path = os.path.join(RESULTS_DIR, run_filename)

                with open(full_file_path, 'w', encoding='utf-8', buffering=1) as f_out:
                    f_out.write(f"experiment run details:\n")
                    f_out.write(f"  dataset: {dataset}\n")
                    f_out.write(f"  path: {dataset_path}\n")
                    f_out.write(f"  similarity location: {layer_name}\n")
                    f_out.write(f"  prompt structure: {prompt_structure_name}\n\n")
                    f_out.write(f"---------------------------------------\n")

                    for test_split_name, test_split in test_splits:
                        f_out.write(f"  test split name: {test_split_name}\n")
                        f_out.write(f"---------------------------------------\n\n")
                        results = {"correct": {"cat_1": 0, "cat_2": 0}, "incorrect": {"cat_1": 0, "cat_2": 0}}

                        for i, item in enumerate(test_split):
                            flip = {"cat_2": "cat_1", "cat_1": "cat_2"}
                            true = item["imagefiles"][item["testfiles"]["category"]]
                            false = item["imagefiles"][flip[item["testfiles"]["category"]]]
                            test = [item["testfiles"]["testimage"]]
                            test_cat = item["testfiles"]["category"]

                            pairs = [(test, []), (true, []), (false, [])]

                            if batched:
                                for i, o in pairs:
                                    images = resize_images([Image.open(dataset_path + url).convert("RGB") for url in i], longest_edge=224)
                                    inputs = processor(images=images, text=prompt(len(images)), return_tensors="pt").to(llm.dtype)
                                    if force_cuda:
                                        inputs = inputs.to("cuda")
                                    cfg = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True)
                                    outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id,
                                                           return_dict_in_generate=True, output_hidden_states=True, generation_config=cfg)
                                    states = outputs["hidden_states"]
                                    o.append(states[0][layer][0][0 + skip_tokens_start:-1 - skip_tokens_end])

                                if evaluate_pairs_b(pairs, True, f_out=f_out):
                                    results["correct"][test_cat] += 1
                                else:
                                    results["incorrect"][test_cat] += 1
                            else:
                                for i, o in pairs:
                                    for item in i:
                                        images = resize_images([Image.open(dataset_path + item).convert("RGB")], longest_edge=224)
                                        inputs = processor(images=images, text=prompt, return_tensors="pt").to(llm.dtype)
                                        if force_cuda:
                                            inputs = inputs.to("cuda")
                                        cfg = GenerationConfig(output_hidden_states=True, return_dict_in_generate=True)
                                        outputs = llm.generate(**inputs, temperature=0, do_sample=False, max_new_tokens=1, eos_token_id=processor.tokenizer.eos_token_id,
                                                               return_dict_in_generate=True, output_hidden_states=True, generation_config=cfg)
                                        states = outputs["hidden_states"]
                                        o.append(states[0][layer][0][skip_tokens_start:-1 - skip_tokens_end])

                                if evaluate_pairs_s(pairs, True, f_out=f_out):
                                    results["correct"][test_cat] += 1
                                else:
                                    results["incorrect"][test_cat] += 1

                        correct = results["correct"]["cat_1"] + results["correct"]["cat_2"]
                        incorrect = results["incorrect"]["cat_1"] + results["incorrect"]["cat_2"]
                        accuracy = correct / max(1, correct + incorrect) * 100
                        f_out.write(f"---------------------------------------\n")
                        f_out.write(f"Summary for Split '{test_split_name}':\n")
                        f_out.write(f" results: {results}\n")
                        f_out.write(f" accuracy: {accuracy:.2f}%\n\n")
                        f_out.write(f"---------------------------------------\n")
