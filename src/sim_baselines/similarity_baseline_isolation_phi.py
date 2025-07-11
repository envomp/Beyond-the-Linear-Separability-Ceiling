from scripts.conf import *
from scripts.hf_models import load_phi_3_5_vision
from similarity_baseline import run_similarity_baseline_isolation

llm, processor = load_phi_3_5_vision()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1)
]

prompts_config = [
    (2, 2, False, "<|user|><|image_1|><|end|><|assistant|>", "single"),
    (2, 2, True, lambda x: f"<|user|>{"".join([f"\n<|image_{i}|>" for i in range(1, x + 1)])}<|end|><|assistant|>", "batched"),
]

run_similarity_baseline_isolation(llm, processor, "phi", datasets, similarity_layers, prompts_config, force_cuda=True)
