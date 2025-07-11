import torch

from scripts.model_modification import TrainableEmbedding, TrainableImage, materialize_lora_weights, apply_lora
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModel, AutoConfig
from torch.utils.checkpoint import checkpoint
from PIL import Image


def load_weights(llm, param_data, param_name=None, no_vision=False):
    for name, param in llm.named_parameters():
        if "vision" in name and no_vision:
            continue
        if isinstance(param_data, dict):
            for k, v in param_data.items():
                if name == k and param.data.shape == v.shape:
                    param.data.copy_(v)
                    print(f"Successfully loaded parameter '{k}' of shape '{v.shape}' with data {v.flatten()[:5]} ...")
                    break
        elif isinstance(param_data, torch.Tensor) and param_name is not None:
            if name == param_name:
                param.data.copy_(param_data)
                print(f"Successfully loaded parameter '{param_name}' of shape '{param_data.shape}' with data {param_data.flatten()[:5]} ...")
                break


def resize_images(images, longest_edge=224):
    resized_images = []
    for img in images:
        original_width, original_height = img.size
        current_longest_edge = max(original_width, original_height)
        if current_longest_edge < longest_edge:
            resized_images.append(img)
            continue
        scale_ratio = longest_edge / current_longest_edge
        new_width = max(1, int(original_width * scale_ratio))
        new_height = max(1, int(original_height * scale_ratio))
        resized_img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        resized_images.append(resized_img)

    return resized_images


def inference(model, processor, prompt, urls=[], images=[], max_tokens=1000, num_return_sequences=1, resize=False, skip_special=False, skip_input=True, force_cuda=False):
    images = [Image.open(url).convert("RGB") for url in urls] if urls else images
    if resize:
        images = resize_images(images, longest_edge=resize)

    if "gemma" in processor.tokenizer.name_or_path:
        processor.tokenizer.padding_side = "left"
        kwargs = {"padding": "longest", "pad_to_multiple_of": 8}
    else:
        kwargs = {}
    inputs = processor(images=images, text=prompt, return_tensors="pt", **kwargs).to(model.dtype)
    if force_cuda:
        inputs = inputs.to("cuda")
    cfg = GenerationConfig(output_hidden_states=False, return_dict_in_generate=True, num_return_sequences=num_return_sequences)

    generation_args = {
        "max_new_tokens": max_tokens,
        "temperature": 0.0,
        "do_sample": False,
    }
    if model.config.eos_token_id:
        eos_token_ids = []
        if not isinstance(model.config.eos_token_id, list):
            eos_token_ids.append(model.config.eos_token_id)
        else:
            eos_token_ids.extend(model.config.eos_token_id)
        eos_token_ids.append(processor.tokenizer.eos_token_id)
    else:
        eos_token_ids = processor.tokenizer.eos_token_id
    outputs = model.generate(**inputs, eos_token_id=eos_token_ids, generation_config=cfg, **generation_args)
    ids = outputs.sequences[:, inputs['input_ids'].shape[1]:] if skip_input else outputs.sequences
    return processor.batch_decode(ids, skip_special_tokens=skip_special)


def raw_forward(model, processor, urls, prompt, resize=False):
    images = [Image.open(url).convert("RGB") for url in urls]
    if resize:
        images = resize_images(images, longest_edge=resize)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to(model.dtype).to("cuda")
    return model(**inputs).logits


def find_image_token_ranges_phi(inputs):
    image_ranges = []

    for sample in inputs['input_ids']:
        image_indices = {}
        current_image_id = None
        start_index = None

        for i, token_id in enumerate(sample):
            if hasattr(token_id, 'item'):
                token_id = token_id.item()

            if token_id < 0:
                if current_image_id is None:
                    current_image_id = token_id
                    start_index = i
                elif token_id != current_image_id:
                    image_indices[-current_image_id] = (start_index, i - 1)
                    current_image_id = token_id
                    start_index = i
            else:
                if current_image_id is not None:
                    image_indices[-current_image_id] = (start_index, i - 1)
                    current_image_id = None
                    start_index = None

        if current_image_id is not None:
            image_indices[-current_image_id] = (start_index, len(inputs) - 1)

        image_ranges.append(image_indices)

    return image_ranges


def find_image_token_ranges_pixtral(inputs):
    image_ranges = []

    for sample in inputs['input_ids']:
        image_indices = {}
        start_index = None
        image_id = 1
        for i, token_id in enumerate(sample):
            if hasattr(token_id, 'item'):
                token_id = token_id.item()

            if token_id == 10 or token_id == 12:
                if start_index is None:
                    start_index = i
            elif token_id == 13:
                if start_index is not None:
                    end_index = i
                    if start_index <= end_index:
                        image_indices[image_id] = (start_index, end_index)
                        image_id += 1
                    start_index = None
            else:
                start_index = None
        image_ranges.append(image_indices)

    return image_ranges


def find_image_token_ranges_gemma(inputs):
    image_ranges = []

    for sample in inputs['input_ids']:
        image_indices = {}
        start_index = None
        image_id = 1
        for i, token_id in enumerate(sample):
            if hasattr(token_id, 'item'):
                token_id = token_id.item()

            if token_id == 255999 or token_id == 262144:
                if start_index is None:
                    start_index = i
            elif token_id == 256000:
                if start_index is not None:
                    end_index = i
                    if start_index <= end_index:
                        image_indices[image_id] = (start_index, end_index)
                        image_id += 1
                    start_index = None
            else:
                start_index = None
        image_ranges.append(image_indices)

    return image_ranges


def find_image_token_ranges_intern(inputs):
    image_ranges = []

    for i in range(len(inputs['input_ids'])):
        sample = inputs['input_ids'][i]
        spans = inputs.image_patch_sizes[i]
        image_indices = {}
        start_index = None
        image_id = 1
        span = None

        for i, token_id in enumerate(sample):
            if hasattr(token_id, 'item'):
                token_id = token_id.item()

            if token_id == 151667:
                if start_index is None:
                    start_index = i
                    span = spans.pop(0) - 1
                else:
                    span -= 1
                    if span == 0:
                        end_index = i
                        image_indices[image_id] = (start_index, end_index)
                        image_id += 1
                        start_index = None
            elif token_id == 151666:
                assert span == 0
                if start_index is not None:
                    end_index = i
                    if start_index <= end_index:
                        image_indices[image_id] = (start_index, end_index)
                        image_id += 1
                    start_index = None
            else:
                start_index = None
        image_ranges.append(image_indices)

    return image_ranges


def find_image_token_ranges_qwen(inputs):
    image_ranges = []

    for sample in inputs['input_ids']:
        image_indices = {}
        start_index = None
        image_id = 1
        for i, token_id in enumerate(sample):
            if hasattr(token_id, 'item'):
                token_id = token_id.item()

            if token_id == 151652 or token_id == 151655 or token_id == 151654:
                if start_index is None:
                    start_index = i
            elif token_id == 151653:
                if start_index is not None:
                    end_index = i
                    if start_index <= end_index:
                        image_indices[image_id] = (start_index, end_index)
                        image_id += 1
                    start_index = None
            else:
                start_index = None
        image_ranges.append(image_indices)

    return image_ranges

def reentrant_checkpoint(f, *args):
    return checkpoint(f, *args, use_reentrant=True)


def split_model(model_path):
    import math

    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def load_intern_vl3_14B():
    from scripts.intern_vl_utils import InternVLProcessor

    name = "OpenGVLab/InternVL3-14B"
    revision = "af63fb67e5b5696ad1ff2b7862c6a1da193c3ebd"

    device_map = split_model(name)
    intern = AutoModel.from_pretrained(
        name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map)

    for param in intern.parameters():
        param.requires_grad = False

    processor = InternVLProcessor(AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision))
    intern.img_context_token_id = processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    print("Listing all trainable parameters:")
    for name, param in intern.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return intern, processor


def load_intern_vl3_78B():
    from scripts.intern_vl_utils import InternVLProcessor

    name = "OpenGVLab/InternVL3-78B"
    revision = "852965ee1490adbc4f0142b8106a7d7772f55fd5"

    device_map = split_model(name)
    intern = AutoModel.from_pretrained(
        name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map)

    for param in intern.parameters():
        param.requires_grad = False

    processor = InternVLProcessor(AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision))
    intern.img_context_token_id = processor.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    print("Listing all trainable parameters:")
    for name, param in intern.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return intern, processor


def load_qwen25_VL_7B():
    from transformers import Qwen2_5_VLForConditionalGeneration

    name = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision = "cc594898137f460bfe9f0759e9844b3ce807cfb5"

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(name,
                                                              revision=revision,
                                                              attn_implementation="flash_attention_2",
                                                              torch_dtype=torch.bfloat16,
                                                              device_map="auto")

    for param in qwen.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in qwen.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return qwen, processor


def load_qwen25_VL_72B():
    from transformers import Qwen2_5_VLForConditionalGeneration

    name = "Qwen/Qwen2.5-VL-72B-Instruct"
    revision = "89c86200743eec961a297729e7990e8f2ddbc4c5"

    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(name,
                                                              revision=revision,
                                                              attn_implementation="flash_attention_2",
                                                              torch_dtype=torch.bfloat16,
                                                              device_map="auto")

    for param in qwen.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in qwen.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return qwen, processor


def load_phi_3_5_vision(trainable_prompt_size=0, trainable_image_size=0, do_checkpoint=False, attn="flash_attention_2", num_crops=4, post_dispatch=lambda x: x):
    from accelerate import load_checkpoint_and_dispatch, init_empty_weights

    name = "microsoft/Phi-3.5-vision-instruct"
    revision = "dfc36d84e079f4d99c654f1cc489e0970f5917c0"
    # path = f"/gpfs/mariana/home/envomp/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct/snapshots/{revision}"
    path = f"/media/e/data/huggingface/hub/models--microsoft--Phi-3.5-vision-instruct/snapshots/{revision}"

    if do_checkpoint:
        with init_empty_weights():
            phi = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, revision=revision, _attn_implementation=attn)

        phi = load_checkpoint_and_dispatch(
            phi,
            path,
            device_map="auto",
            no_split_module_classes=[],
            max_memory={0: "20GiB", "cpu": "100GiB"},
            dtype=torch.bfloat16
        )
        for param in phi.parameters():
            param.requires_grad = False
        phi = post_dispatch(phi)

        phi.model.gradient_checkpointing = True
        phi.model._gradient_checkpointing_func = reentrant_checkpoint
    else:
        phi = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, revision=revision, _attn_implementation=attn, torch_dtype=torch.bfloat16, device_map="cuda")
        phi = post_dispatch(phi)
        for param in phi.parameters():
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision, num_crops=num_crops)

    if trainable_prompt_size:
        embedding_layer = phi.model.vision_embed_tokens
        phi.model.vision_embed_tokens = TrainableEmbedding(embedding_layer, dim=3072, trainable_prompt_size=trainable_prompt_size)

    if trainable_image_size:
        if do_checkpoint:
            phi.model.vision_embed_tokens.img_processor.vision_model.encoder.gradient_checkpointing = True
            phi.model.vision_embed_tokens.img_processor.vision_model.encoder._gradient_checkpointing_func = reentrant_checkpoint
        embedding_layer = phi.model.vision_embed_tokens.img_processor
        phi.model.vision_embed_tokens.img_processor = TrainableImage(embedding_layer, image_size=trainable_image_size, dim=trainable_image_size)

    print("Listing all trainable parameters:")
    for name, param in phi.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return phi, processor


def lora_post_dispatch(llm):
    proj_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
    llm.enable_input_require_grads()
    llm = apply_lora(llm, target_modules=proj_keywords)  # attention & ffn proj for vision and language stack
    llm = materialize_lora_weights(llm)
    return llm


def load_pixtral_12B(trainable_prompt_size=0, do_checkpoint=False, post_dispatch=lambda x: x):
    from accelerate import cpu_offload
    from transformers import LlavaForConditionalGeneration

    name = "mistral-community/pixtral-12b"
    revision = "c2756cbbb9422eba9f6c5c439a214b0392dfc998"

    if do_checkpoint:
        pixtral = LlavaForConditionalGeneration.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16, device_map="cpu")
        pixtral = cpu_offload(pixtral, execution_device="cuda:0")
        for param in pixtral.parameters():
            param.requires_grad = False
        pixtral = post_dispatch(pixtral)

        pixtral.language_model.gradient_checkpointing = True
        pixtral.language_model._gradient_checkpointing_func = reentrant_checkpoint
        pixtral.vision_tower.gradient_checkpointing = True
        pixtral.vision_tower._gradient_checkpointing_func = reentrant_checkpoint
    else:
        pixtral = LlavaForConditionalGeneration.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda:0")
        pixtral = post_dispatch(pixtral)
        for param in pixtral.parameters():
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision)

    if trainable_prompt_size:
        embedding_layer = pixtral.language_model.get_input_embeddings()
        pixtral.language_model.set_input_embeddings(TrainableEmbedding(embedding_layer, dim=5120, trainable_prompt_size=trainable_prompt_size))

    print("Listing all trainable parameters:")
    for name, param in pixtral.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return pixtral, processor


def load_gemma3_27B():
    from transformers import Gemma3ForConditionalGeneration

    name = "google/gemma-3-27b-it"
    revision = "005ad3404e59d6023443cb575daa05336842228a"

    gemma = Gemma3ForConditionalGeneration.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16, device_map="auto")
    for param in gemma.parameters():
        param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision)

    print("Listing all trainable parameters:")
    for name, param in gemma.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return gemma, processor


def load_gemma3_4B(trainable_prompt_size=0, do_checkpoint=False, post_dispatch=lambda x: x):
    from transformers import Gemma3ForConditionalGeneration

    name = "google/gemma-3-4b-it"
    revision = "093f9f388b31de276ce2de164bdc2081324b9767"

    if do_checkpoint:
        gemma = Gemma3ForConditionalGeneration.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="eager")
        for param in gemma.parameters():
            param.requires_grad = False
        gemma = post_dispatch(gemma)

        gemma._set_gradient_checkpointing(True)
        gemma._gradient_checkpointing_func = reentrant_checkpoint
    else:
        gemma = Gemma3ForConditionalGeneration.from_pretrained(name, revision=revision, torch_dtype=torch.bfloat16, device_map="cuda:0")
        gemma = post_dispatch(gemma)
        for param in gemma.parameters():
            param.requires_grad = False

    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True, revision=revision)

    if trainable_prompt_size:
        embedding_layer = gemma.language_model.get_input_embeddings()
        gemma.language_model.set_input_embeddings(TrainableEmbedding(embedding_layer, dim=2560, trainable_prompt_size=trainable_prompt_size, replace_token_id=3))

    print("Listing all trainable parameters:")
    for name, param in gemma.named_parameters():
        if param.requires_grad:
            print(f"> {name}, dtype={param.dtype}, device={param.device}")
    return gemma, processor
