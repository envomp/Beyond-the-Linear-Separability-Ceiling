from scripts.conf import *
from src.processor import interleaved_test_first_prompt, interleaved_prompt, labeled_test_first_prompt, labeled_prompt, construct_prompt_phi
from similarity_baseline import run_similarity_baselines
from scripts.hf_models import load_phi_3_5_vision, find_image_token_ranges_phi

phi, processor = load_phi_3_5_vision()

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

run_similarity_baselines(phi, processor, "phi", datasets, similarity_layers, prompts_config, find_image_token_ranges_phi, construct_prompt_phi, force_cuda=True)
