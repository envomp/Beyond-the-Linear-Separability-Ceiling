from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, direct_prompt, cot_prompt, labeled_prompt, construct_prompt_phi
from baseline import run_baselines

phi, processor = load_phi_3_5_vision()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

prompt_method_configs = [
    ("direct", direct_prompt),
    ("cot", cot_prompt)
]

prompts_config = [
    (interleaved_prompt, False, "interleaved"),
    (interleaved_test_first_prompt, True, "interleaved_test_first"),
    (labeled_prompt, False, "labeled"),
    (labeled_test_first_prompt, True, "labeled_test_first")
]

run_baselines(phi, processor, "phi", datasets, prompt_method_configs, prompts_config, construct_prompt_phi, force_cuda=True)
