import torch

from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, inference, lora_post_dispatch, load_weights, load_pixtral_12B, load_gemma3_4B, get_noise_injection_hook
from datasets import load_dataset
from tqdm import tqdm
from src.best_PEFT import c_scan_phi_loras

def eval_image_retrieval(model, processor, image_tags, prompt_template, log_fn=print):
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
    log_fn("\n--- Evaluation Results (Image Retrieval) ---")
    log_fn(f"\nImage retrieval score accuracy: {image_retrieval_accuracy:.2f}% ({image_retrieval_correct}/{total_examples})")
    log_fn(image_retrieval_results)

def eval_text_retrieval(model, processor, image_tag, prompt_template, log_fn=print):
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
    log_fn("\n--- Evaluation Results (Text Retrieval) ---")
    log_fn(f"\nText retrieval score accuracy: {text_retrieval_accuracy:.2f}% ({text_retrieval_correct}/{total_examples})")
    log_fn(text_retrieval_results)

add_noise = False
# loras = {
#     ("phi", None, None, False): None,
#     ("phi", "lora", "hoi", False): "sim_hoi_phi_lora_c_0.0_e_2_t_0_acc_74_81_77_82_seed_9712.pt",
#     ("phi", "lora", "hoi", True): "sim_hoi_phi_lora_c_0.4_e_2_t_0_acc_79_80_81_82_seed_4645.pt",
#     ("phi", "lora", "openworld", False): "sim_openworld_phi_lora_c_0.0_e_19_t_0_acc_93_seed_3625.pt",
#     ("phi", "lora", "openworld", True): "sim_openworld_phi_lora_c_0.4_e_6_t_0_acc_99_seed_6592.pt",
#     ("pixtral", None, None, False): None,
#     ("pixtral", "lora", "hoi", False): "sim_hoi_pixtral_lora_c_0.0_e_1_t_0_acc_74_75_79_80_seed_5567.pt",
#     ("pixtral", "lora", "hoi", True): "sim_hoi_pixtral_lora_c_1.6_e_2_t_0_acc_74_77_83_75_seed_5899.pt",
#     ("pixtral", "lora", "openworld", False): "sim_openworld_pixtral_lora_c_0.0_e_0_t_0_acc_96_seed_5964.pt",
#     ("pixtral", "lora", "openworld", True): "sim_openworld_pixtral_lora_c_1.6_e_19_t_0_acc_98_seed_7245.pt",
#     ("gemma3_4b", None, None, None): None,
#     ("gemma3_4b", "lora", "hoi", False): "sim_hoi_gemma3_4b_lora_c_0.0_e_2_t_0_acc_84_80_87_86_seed_6563.pt",
#     ("gemma3_4b", "lora", "hoi", True): "sim_hoi_gemma3_4b_lora_c_0.4_e_2_t_0_acc_87_75_82_80_seed_5375.pt",
#     ("gemma3_4b", "lora", "openworld", False): "sim_openworld_gemma3_4b_lora_c_0.0_e_18_t_0_acc_95_seed_9188.pt",
#     ("gemma3_4b", "lora", "openworld", True): "sim_openworld_gemma3_4b_lora_c_0.4_e_19_t_0_acc_99_seed_13.pt",
# }

# loras = c_scan_phi_loras

loras, add_noise = {
    ("phi", None, None, None): None,
    ("phi", "lora", "hoi", False): "sim_hoi_phi_lora_c_0.0_e_2_t_0_acc_74_81_77_82_seed_9712.pt",
    ("phi", "lora", "hoi", True): "sim_hoi_phi_lora_c_0.4_e_2_t_0_acc_79_80_81_82_seed_4645.pt",
}, True # noise experiment

run_filename = f"results_winoground.txt"
with open(run_filename, 'a', encoding='utf-8', buffering=1) as f_out:
    file_logger = lambda *args: f_out.write(" ".join(map(str, args)) + "\n")

    for model, loc, ds, sim in loras:
        if model == "phi":
            prompt_template = lambda user, assistant: f"<|user|> {user} <|end|>\n<|assistant|>\n{assistant}"
            image_tags = ["<|image_1|>", "<|image_2|>"]
            load_model = load_phi_3_5_vision
            vision_encoder = lambda llm: llm.model.model.vision_embed_tokens.img_projection if loc == "lora" else llm.model.vision_embed_tokens.img_projection
        elif model == "pixtral":
            prompt_template = lambda user, assistant: f"<s>[INST] {user} [/INST]\n{assistant}"
            image_tags = ["[IMG]", "[IMG]"]
            load_model = load_pixtral_12B
            vision_encoder = lambda llm: llm.model.multi_modal_projector if loc == "lora" else llm.multi_modal_projector
        elif model == "gemma3_4b":
            prompt_template = lambda user, assistant: f"<bos><start_of_turn>user\n{user} <end_of_turn>\n<start_of_turn>model\n{assistant}"
            image_tags = ["<start_of_image>", "<start_of_image>"]
            load_model = load_gemma3_4B
            vision_encoder = lambda llm: llm.model.multi_modal_projector if loc == "lora" else llm.multi_modal_projector
        else:
            raise RuntimeError("unknown model: ", model)

        file_logger(f"\n\n model: {model}, loc: {loc}, train_ds: {ds}, train_obj_sim: {sim}")

        if loc == "lora":
            llm, processor = load_model(post_dispatch=lora_post_dispatch)
            param_data = torch.load(PEFT_PATH + loras[(model, loc, ds, sim)], weights_only=True)
            load_weights(llm.eval(), param_data, no_vision=True)
        else:
            llm, processor = load_model()
        llm = llm.eval()

        if add_noise:
            for noise in range(0, 11, 2):
                noise /= 10
                t_scores, i_scores = {}, {}
                t_dict_logger = lambda *args: t_scores.__setitem__(noise, args)
                i_dict_logger = lambda *args: i_scores.__setitem__(noise, args)
                handle = vision_encoder(llm).register_forward_hook(get_noise_injection_hook(noise_level=noise))
                eval_text_retrieval(llm, processor, image_tag=image_tags[0], prompt_template=prompt_template, log_fn=t_dict_logger)
                file_logger("t_noise_scores:", t_scores)
                # eval_image_retrieval(llm, processor, image_tags=image_tags, prompt_template=prompt_template, log_fn=i_dict_logger)
                # file_logger("i_noise_scores:", i_scores)
                handle.remove()
        else:
            eval_text_retrieval(llm, processor, image_tag=image_tags[0], prompt_template=prompt_template, log_fn=file_logger)
            eval_image_retrieval(llm, processor, image_tags=image_tags, prompt_template=prompt_template, log_fn=file_logger)

        del llm, processor
