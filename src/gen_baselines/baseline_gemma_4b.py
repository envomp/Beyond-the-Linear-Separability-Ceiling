from scripts.conf import *
from scripts.hf_models import load_gemma3_27B, load_gemma3_4B
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, direct_prompt, cot_prompt, labeled_prompt, construct_prompt_gemma
from baseline import run_baselines

gemma, processor = load_gemma3_4B()

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

run_baselines(gemma, processor, "gemma_4b", datasets, prompt_method_configs, prompts_config, construct_prompt_gemma, force_cuda=True)
