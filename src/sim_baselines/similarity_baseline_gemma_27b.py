from scripts.conf import *
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, labeled_prompt, construct_prompt_gemma
from similarity_baseline import run_similarity_baselines
from scripts.hf_models import load_gemma3_27B, find_image_token_ranges_gemma

pixtral, processor = load_gemma3_27B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1)
]

prompts_config = [
    (interleaved_prompt, False, "interleaved"),
    (interleaved_test_first_prompt, True, "interleaved_test_first"),
    (labeled_prompt, False, "labeled"),
    (labeled_test_first_prompt, True, "labeled_test_first")
]

run_similarity_baselines(pixtral, processor, "gemma_27b", datasets, similarity_layers, prompts_config, find_image_token_ranges_gemma, construct_prompt_gemma)
