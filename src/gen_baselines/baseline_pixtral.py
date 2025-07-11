from scripts.conf import *
from scripts.hf_models import load_pixtral_12B
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, direct_prompt, cot_prompt, labeled_prompt, construct_prompt_pixtral
from baseline import run_baselines

pixtral, processor = load_pixtral_12B()

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

run_baselines(pixtral, processor, "pixtral", datasets, prompt_method_configs, prompts_config, construct_prompt_pixtral, force_cuda=True)
