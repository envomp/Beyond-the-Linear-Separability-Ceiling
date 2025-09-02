import torch

from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, inference, lora_post_dispatch, load_weights, load_pixtral_12B, load_gemma3_4B
from datasets import load_dataset
from tqdm import tqdm


def eval_image_retrieval(model, processor, image_tags, prompt_template):
    dataset = load_dataset("facebook/winoground", split="test")
    image_retrieval_correct = 0
    image_retrieval_results = {}
    total_examples = len(dataset)
    for example in tqdm(dataset, desc="Evaluating Image Retrieval"):
        image_0 = example['image_0'].convert("RGB")
        image_1 = example['image_1'].convert("RGB")
        caption_0 = example['caption_0']
        caption_1 = example['caption_1']
        tag = example['collapsed_tag']
        images = [image_0, image_1]

        eval_f_0 = lambda x: "cat_2" in x and "cat_1" not in x
        eval_f_1 = lambda x: "cat_1" in x and "cat_2" not in x
        correct_list = []
        for caption, eval_f in [(caption_0, eval_f_0), (caption_1, eval_f_1)]:
            task = (f"You are given a rule and two images, labeled 'cat_2' and 'cat_1'.\n"
                    f"Rule: \"{caption}\"\n"
                    f"Image cat_2: {image_tags[0]}\n"
                    f"Image cat_1: {image_tags[1]}\n"
                    f"Which image does the rule describe? Respond with only the label 'cat_2' or 'cat_1'.")
            prompt = prompt_template(task, "Conclusion:")
            response = inference(model=model, processor=processor, prompt=prompt, images=images, max_tokens=10, skip_special=True, force_cuda=True)
            correct_list.append(eval_f(response[0]))

        if all(correct_list):
            image_retrieval_correct += 1
            if not image_retrieval_results.get(tag):
                image_retrieval_results[tag] = 0
            image_retrieval_results[tag] += 1

    image_retrieval_accuracy = (image_retrieval_correct / total_examples) * 100 if total_examples > 0 else 0
    print("--- Evaluation Results (Image Retrieval) ---")
    print(f"\nImage retrieval score accuracy: {image_retrieval_accuracy:.2f}% ({image_retrieval_correct}/{total_examples})")
    print(image_retrieval_results, "\n")


def eval_text_retrieval(model, processor, image_tag, prompt_template):
    dataset = load_dataset("facebook/winoground", split="test")
    text_retrieval_correct = 0
    text_retrieval_results = {}
    total_examples = len(dataset)
    for example in tqdm(dataset, desc="Evaluating Text Retrieval"):
        image_0 = example['image_0'].convert("RGB")
        image_1 = example['image_1'].convert("RGB")
        caption_0 = example['caption_0']
        caption_1 = example['caption_1']
        tag = example['collapsed_tag']

        eval_f_0 = lambda x: "cat_2" in x and "cat_1" not in x
        eval_f_1 = lambda x: "cat_1" in x and "cat_2" not in x
        correct_list = []
        for image, eval_f in [(image_0, eval_f_0), (image_1, eval_f_1)]:
            task = (f"You are given an image and two rules, labeled 'cat_2' and 'cat_1'. \n"
                    f"Image: {image_tag}\n"
                    f"Rule cat_2: {caption_0} \n"
                    f"Rule cat_1: {caption_1} \n"
                    f"Your task is to:\n"
                    f"Which caption better describes the image? Respond with only the label 'cat_2' or 'cat_1'.")
            prompt = prompt_template(task, "Conclusion:")

            response = inference(model=model, processor=processor, prompt=prompt, images=[image], max_tokens=10, skip_special=True, force_cuda=True)
            correct_list.append(eval_f(response[0]))

        if all(correct_list):
            text_retrieval_correct += 1
            if not text_retrieval_results.get(tag):
                text_retrieval_results[tag] = 0
            text_retrieval_results[tag] += 1

    text_retrieval_accuracy = (text_retrieval_correct / total_examples) * 100 if total_examples > 0 else 0
    print("--- Evaluation Results (Text Retrieval) ---")
    print(f"\nText retrieval score accuracy: {text_retrieval_accuracy:.2f}% ({text_retrieval_correct}/{total_examples})")
    print(text_retrieval_results, "\n")


loras = {
    ("phi", None, None, False): None,
    ("phi", "lora", "hoi", False): "sim_hoi_phi_lora_c_0.0_e_2_t_0_acc_74_81_77_82_seed_9712.pt",
    ("phi", "lora", "hoi", True): "sim_hoi_phi_lora_c_0.4_e_2_t_0_acc_79_80_81_82_seed_4645.pt",
    ("phi", "lora", "openworld", False): "sim_openworld_phi_lora_c_0.0_e_19_t_0_acc_93_seed_3625.pt",
    ("phi", "lora", "openworld", True): "sim_openworld_phi_lora_c_0.4_e_6_t_0_acc_99_seed_6592.pt",
    ("pixtral", None, None, False): None,
    ("pixtral", "lora", "hoi", False): "sim_hoi_pixtral_lora_c_0.0_e_1_t_0_acc_74_75_79_80_seed_5567.pt",
    ("pixtral", "lora", "hoi", True): "sim_hoi_pixtral_lora_c_1.6_e_2_t_0_acc_74_77_83_75_seed_5899.pt",
    ("pixtral", "lora", "openworld", False): "sim_openworld_pixtral_lora_c_0.0_e_0_t_0_acc_96_seed_5964.pt",
    ("pixtral", "lora", "openworld", True): "sim_openworld_pixtral_lora_c_1.6_e_19_t_0_acc_98_seed_7245.pt",
    ("gemma3_4b", None, None, None): None,
    ("gemma3_4b", "lora", "hoi", False): "sim_hoi_gemma3_4b_lora_c_0.0_e_2_t_0_acc_84_80_87_86_seed_6563.pt",
    ("gemma3_4b", "lora", "hoi", True): "sim_hoi_gemma3_4b_lora_c_0.4_e_2_t_0_acc_87_75_82_80_seed_5375.pt",
    ("gemma3_4b", "lora", "openworld", False): "sim_openworld_gemma3_4b_lora_c_0.0_e_18_t_0_acc_95_seed_9188.pt",
    ("gemma3_4b", "lora", "openworld", True): "sim_openworld_gemma3_4b_lora_c_0.4_e_19_t_0_acc_99_seed_13.pt",
}

for model, loc, ds, sim in loras:
    if model == "phi":
        prompt_template = lambda user, assistant: f"<|user|> {user} <|end|>\n<|assistant|>\n{assistant}"
        image_tags = ["<|image_1|>", "<|image_2|>"]
        load_model = load_phi_3_5_vision
    elif model == "pixtral":
        prompt_template = lambda user, assistant: f"<s>[INST] {user} [/INST]\n{assistant}"
        image_tags = ["[IMG]", "[IMG]"]
        load_model = load_pixtral_12B
    elif model == "gemma3_4b":
        prompt_template = lambda user, assistant: f"<bos><start_of_turn>user\n{user} <end_of_turn>\n<start_of_turn>model\n{assistant}"
        image_tags = ["<start_of_image>", "<start_of_image>"]
        load_model = load_gemma3_4B
    else:
        raise RuntimeError("unknown model: ", model)

    print(f"\n\n model: {model}, loc: {loc}, train_ds: {ds}, train_obj_sim: {sim}")
    if loc == "lora":
        llm, processor = load_model(post_dispatch=lora_post_dispatch)
        print(llm)
        param_data = torch.load(PEFT_PATH + loras[(model, loc, ds, sim)], weights_only=True)
        load_weights(llm.eval(), param_data, no_vision=True)
        eval_text_retrieval(llm, processor, image_tag=image_tags[0], prompt_template=prompt_template)
        eval_image_retrieval(llm, processor, image_tags=image_tags, prompt_template=prompt_template)
    else:
        llm, processor = load_model()
        eval_text_retrieval(llm.eval(), processor, image_tag=image_tags[0], prompt_template=prompt_template)
        eval_image_retrieval(llm.eval(), processor, image_tags=image_tags, prompt_template=prompt_template)
    del llm, processor
