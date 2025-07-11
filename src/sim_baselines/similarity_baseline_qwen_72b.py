from scripts.conf import *
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, labeled_prompt, construct_prompt_qwen
from similarity_baseline import run_similarity_baselines
from scripts.hf_models import load_qwen25_VL_72B, find_image_token_ranges_qwen

qwen, processor = load_qwen25_VL_72B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1),
]

prompts_config = [
    (interleaved_prompt, False, "interleaved"),
    (interleaved_test_first_prompt, True, "interleaved_test_first"),
    (labeled_prompt, False, "labeled"),
    (labeled_test_first_prompt, True, "labeled_test_first")
]


run_similarity_baselines(qwen, processor, "qwen_72b", datasets, similarity_layers, prompts_config, find_image_token_ranges_qwen, construct_prompt_qwen)
