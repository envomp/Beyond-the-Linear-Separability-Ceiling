from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision, lora_post_dispatch, load_weights
from baseline import run_vqa_baselines, load_aokvqa_data, load_hoi_prototype_data, load_scienceqa_data, load_pope_data
from processor import construct_prompt_phi
import torch

c_scan_phi_loras = {
    ("phi", "baseline"): None,
    ("phi", "hoi_nt"): "sim_hoi_phi_lora_c_0.0_e_2_t_0_acc_74_81_77_82_seed_9712.pt",
    ("phi", "hoi_c"): "sim_hoi_phi_lora_c_0.4_e_2_t_0_acc_79_80_81_82_seed_4645.pt",
}

DATASET_TO_RUN = "hoi"  # aokvqa or hoi or scienceqa or pope

for model, method in c_scan_phi_loras:
    post_dispatch = lambda x: lora_post_dispatch(x, ignore_vision=True)
    llm, processor = load_phi_3_5_vision(post_dispatch=post_dispatch if method != "baseline" else lambda x: x)
    if method != "baseline":
        param_data = torch.load(PEFT_PATH + c_scan_phi_loras[(model, method)], weights_only=True)
        load_weights(llm.eval(), param_data, no_vision=True)
    prompt = lambda x, y: construct_prompt_phi(x, y, postfix="Answer:")

    if DATASET_TO_RUN == "aokvqa":
        vqa_dataset = load_aokvqa_data(split="validation")
    elif DATASET_TO_RUN == "hoi":
        vqa_dataset = load_hoi_prototype_data("bongard_hoi_vqa.json", HOI_DATASET_PATH, split_prefix="test")[0]
    elif DATASET_TO_RUN == "scienceqa":
        vqa_dataset = load_scienceqa_data(split="test")
    elif DATASET_TO_RUN == "pope":
        vqa_dataset = load_pope_data(split="test")
    else:
        raise ValueError(f"Unknown DATASET_TO_RUN: {DATASET_TO_RUN}")

    run_vqa_baselines(llm, processor, "phi", vqa_dataset, prompt, run_prefix=f"eval_{method}_on_{DATASET_TO_RUN}_")
