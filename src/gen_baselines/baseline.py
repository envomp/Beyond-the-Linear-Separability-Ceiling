import os
from scripts.hf_models import inference
from src.processor import get_ds, get_urls_and_cat_from_item


def extract_conclusion(text: str):
    text_to_search_cats_in = text
    if "Conclusion" in text_to_search_cats_in:
        parts = text_to_search_cats_in.rsplit("Conclusion", 1)
        if len(parts) > 1:
            text_to_search_cats_in = parts[-1]

    index_cat_1 = text_to_search_cats_in.find("cat_1")
    index_cat_2 = text_to_search_cats_in.find("cat_2")

    if index_cat_1 != -1 and index_cat_2 != -1:
        if index_cat_1 < index_cat_2:
            return "cat_1"
        else:
            return "cat_2"

    elif index_cat_1 != -1:
        return "cat_1"
    elif index_cat_2 != -1:
        return "cat_2"
    else:
        return None


def run_baselines(llm, processor, model_str, datasets, prompt_method_configs, prompts_config, get_full_prompt_f, force_cuda=False):
    RESULTS_DIR = f"../../result/{model_str}/baselines"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    llm.eval()
    for dataset, dataset_path in datasets:
        _, _, test = get_ds(dataset)

        for prompt_method_name, prompt_method in prompt_method_configs:
            for prompt_structure, test_first, prompt_structure_name in prompts_config:
                run_filename = f"results_{dataset}_{prompt_method_name}_{prompt_structure_name}.txt"
                full_file_path = os.path.join(RESULTS_DIR, run_filename)

                with open(full_file_path, 'w', encoding='utf-8', buffering=1) as f_out:
                    f_out.write(f"experiment run details:\n")
                    f_out.write(f"  dataset: {dataset}\n")
                    f_out.write(f"  path: {dataset_path}\n")
                    f_out.write(f"  prompt method: {prompt_method_name}\n")
                    f_out.write(f"  prompt structure: {prompt_structure_name}\n\n")
                    f_out.write(f"---------------------------------------\n")

                    for test_split_name, test_split in test:
                        f_out.write(f"  test split name: {test_split_name}\n")
                        f_out.write(f"---------------------------------------\n\n")
                        results = {"correct": {"cat_1": 0, "cat_2": 0}, "incorrect": {"cat_1": 0, "cat_2": 0}}
                        for i, item in enumerate(test_split):
                            image_urls, test_cat = get_urls_and_cat_from_item(item, dataset_path)
                            image_count = len(image_urls)
                            cat_imgs = (image_count - 1) // 2
                            if test_first:
                                image_urls = [image_urls[-1]] + image_urls[:-1]
                            full_prompt = get_full_prompt_f(cat_imgs, image_count, prompt_method, prompt_structure, test_first, prompt_method_name == "direct")
                            skip_input = "intern" not in model_str
                            decoded = inference(model=llm, processor=processor, urls=image_urls, prompt=full_prompt, resize=224, max_tokens=1500, skip_special=True, skip_input=skip_input, force_cuda=force_cuda)
                            conc = extract_conclusion(decoded[0])
                            if conc == test_cat:
                                results["correct"][test_cat] += 1
                            else:
                                results["incorrect"][test_cat] += 1
                            f_out.write(f"{i} | expected:'{test_cat}' | got='{conc}' | full: {decoded}\n")

                        correct = results["correct"]["cat_1"] + results["correct"]["cat_2"]
                        incorrect = results["incorrect"]["cat_1"] + results["incorrect"]["cat_2"]
                        accuracy = correct / max(1, correct + incorrect) * 100
                        f_out.write(f"---------------------------------------\n")
                        f_out.write(f"Summary for Split '{test_split_name}':\n")
                        f_out.write(f" results: {results}\n")
                        f_out.write(f" accuracy: {accuracy:.2f}%\n\n")
                        f_out.write(f"---------------------------------------\n")
