from scripts.conf import *
from scripts.hf_models import load_gemma3_27B
from similarity_baseline import run_similarity_baseline_isolation

llm, processor = load_gemma3_27B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1)
]

prompts_config = [
    (2, 1, False, "<bos><start_of_turn><start_of_image><end_of_turn>", "single"),
    (2, 1, True, lambda x: f"<bos><start_of_turn>{"".join(["<start_of_image>"] * x)}<end_of_turn>", "batched"),
]

run_similarity_baseline_isolation(llm, processor, "gemma_27b", datasets, similarity_layers, prompts_config)
