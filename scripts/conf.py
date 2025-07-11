# This file should be loaded in the start of every notebook
import os

HUGGINGFACE_DIR = "/media/e/data/huggingface"
MODEL_DIR = HUGGINGFACE_DIR + "/hub"

os.environ["HF_HOME"] = HUGGINGFACE_DIR
os.environ["HF_HUB_CACHE"] = MODEL_DIR
os.environ["PYDEVD_USE_FRAME_EVAL"] = "NO"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from huggingface_hub import login

# login(token="hf_...")

HOI_DATASET_PATH = "~/Downloads/hake/"
OPENWORLD_DATASET_PATH = "~/Downloads/images/"
BONGARD_RESULT_DIR = "~/Downloads/bongard_PEFT/"
PEFT_PATH = "~/Downloads/bongard_PEFT/"
