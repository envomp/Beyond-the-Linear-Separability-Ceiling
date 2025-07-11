from scripts.conf import *
from scripts.hf_models import load_qwen25_VL_7B
from similarity_baseline import run_similarity_baseline_isolation

llm, processor = load_qwen25_VL_7B()

datasets = [
    ("hoi", HOI_DATASET_PATH),
    ("openworld", OPENWORLD_DATASET_PATH),
]

similarity_layers = [
    ("vision", 0),
    ("final", -1)
]

prompts_config = [
    (3, 2, False, "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>", "single"),
    (3, 2, True, lambda x: f"<|im_start|>user\n{"".join([f"<|vision_start|><|image_pad|><|vision_end|><|im_end|>" for _ in range(x)])}", "batched"),
]

run_similarity_baseline_isolation(llm, processor, "qwen_7b", datasets, similarity_layers, prompts_config, force_cuda=True)
