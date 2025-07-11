from scripts.conf import *
from scripts.hf_models import load_intern_vl3_14B
from similarity_baseline import run_similarity_baseline_isolation

llm, processor = load_intern_vl3_14B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1)
]

prompts_config = [
    (0, 0, False, "<image>", "single"),
    (0, 0, True, lambda x: f"{"".join([f"<image>" for _ in range(x)])}", "batched"),
]

run_similarity_baseline_isolation(llm, processor, "intern_14b", datasets, similarity_layers, prompts_config)
