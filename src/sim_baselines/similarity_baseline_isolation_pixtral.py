from scripts.conf import *
from similarity_baseline import run_similarity_baseline_isolation
from scripts.hf_models import load_pixtral_12B

llm, processor = load_pixtral_12B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1)
]

prompts_config = [
    (2, 2, False, f"<s>[INST][IMG][/INST]", "single"),
    (2, 2, True, lambda x: f"<s>[INST]{"".join([f"[IMG]"] * x)}[/INST]", "batched"),
]

run_similarity_baseline_isolation(llm, processor, "pixtral", datasets, similarity_layers, prompts_config, force_cuda=True)
