from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, lora_post_dispatch, load_weights
from baseline import run_vqa_baselines, load_gqa_data, load_aokvqa_data
from processor import construct_prompt_phi
import torch

loras = {
    ("phi", "baseline"): None,
    ("phi", "gqa_c_0.4"): "sim_gqa_phi_lora_c_0.4_e_0_acc_8819_seed_941.pt",
    ("phi", "gqa_c_0.1"): "sim_gqa_phi_lora_c_0.1_e_0_acc_9873_seed_952.pt",
    ("phi", "gqa_c_0.01"): "sim_gqa_phi_lora_c_0.01_e_0_acc_9728_seed_1012.pt",
    ("phi", "gqa_nt"): "sim_gqa_phi_lora_c_0.0_e_0_acc_9847_seed_7894.pt",
}

DATASET_TO_RUN = "aokvqa"  # aokvqa or gqa

for model, method in loras:
    post_dispatch = lambda x: lora_post_dispatch(x, ignore_vision=True)
    llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch if method != "baseline" else lambda x: x)
    if method != "baseline":
        param_data = torch.load(PEFT_PATH + loras[(model, method)], weights_only=True)
        load_weights(llm.eval(), param_data, no_vision=True)
    prompt = lambda x, y: construct_prompt_phi(x, y, postfix="Answer:")

    if DATASET_TO_RUN == "gqa":
        vqa_dataset = load_gqa_data("gqa_contrastive_pairs_eval.json", GQA_IMAGE_DIR)
        dataset_name_str = "gqa"
    elif DATASET_TO_RUN == "aokvqa":
        vqa_dataset = load_aokvqa_data(split="validation")
        dataset_name_str = "aokvqa"
    else:
        raise ValueError(f"Unknown DATASET_TO_RUN: {DATASET_TO_RUN}")

    run_vqa_baselines(llm, processor, "phi", vqa_dataset, prompt, run_prefix=f"eval_{method}_on_{dataset_name_str}_")
