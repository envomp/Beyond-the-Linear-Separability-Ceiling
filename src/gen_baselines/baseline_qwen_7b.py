from scripts.conf import *
from scripts.hf_models import load_qwen25_VL_7B
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, direct_prompt, cot_prompt, labeled_prompt, construct_prompt_qwen
from baseline import run_baselines

qwen, processor = load_qwen25_VL_7B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

prompt_method_configs = [
    ("direct", direct_prompt),
    ("cot", cot_prompt),
]

prompts_config = [
    (interleaved_prompt, False, "interleaved"),
    (interleaved_test_first_prompt, True, "interleaved_test_first"),
    (labeled_prompt, False, "labeled"),
    (labeled_test_first_prompt, True, "labeled_test_first")
]

run_baselines(qwen, processor, "qwen_7b", datasets, prompt_method_configs, prompts_config, construct_prompt_qwen, force_cuda=True)
