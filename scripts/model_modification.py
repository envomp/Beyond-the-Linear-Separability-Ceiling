import torch
from torch import nn


def apply_lora(nn_module, rank=8, target_modules=(), dropout=0):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        lora_dropout=dropout)
    return get_peft_model(nn_module, config)


# PEFT library breaks with weights offloaded to meta device
def materialize_lora_weights(llm):
    params_to_materialize = {}
    for name, param in llm.named_parameters():
        if not ("lora_A" in name or "lora_B" in name):
            continue

        new_data = torch.empty(param.shape, dtype=param.dtype, device=torch.device("cuda:0"), requires_grad=True)
        if "lora_B" in name:
            nn.init.zeros_(new_data)
        elif "lora_A" in name:
            nn.init.kaiming_uniform_(new_data, a=5 ** 0.5)
        params_to_materialize[name] = new_data.requires_grad_(True)
    llm.load_state_dict(params_to_materialize, strict=False, assign=True)
    return llm


class TrainableImage(nn.Module):
    def __init__(self, original_embeddings, image_size, dim):
        super().__init__()
        self.original_embeddings = original_embeddings
        self.trainable_embeddings = nn.Parameter(torch.randn(image_size, dim))
        nn.init.xavier_uniform_(self.trainable_embeddings)

    def forward(self, img_embeds: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        combined = img_embeds + self.trainable_embeddings.to(img_embeds)
        return self.original_embeddings(combined, *args, **kwargs)


class TrainableEmbedding(nn.Module):
    def __init__(self, original_embeddings, dim, replace_token_id=0, trainable_prompt_size=0):
        super().__init__()
        self.original_embeddings = original_embeddings
        self.replace_token_id = replace_token_id
        self.trainable_prompt_size = trainable_prompt_size

        self.trainable_embeddings = nn.Parameter(torch.randn(self.trainable_prompt_size, dim, dtype=torch.bfloat16, device="cuda"))
        nn.init.xavier_uniform_(self.trainable_embeddings)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.FloatTensor:
        with torch.no_grad():
            original_embeddings_output = self.original_embeddings(input_ids, *args, **kwargs)

        if input_ids.shape[1] == 1:  # inference mode caching
            return original_embeddings_output

        if not (input_ids == self.replace_token_id).any():  # no trainable tokens detected, normal forward pass
            return original_embeddings_output

        combined_embeddings = original_embeddings_output.clone()
        combined_embeddings[input_ids == self.replace_token_id] = self.trainable_embeddings

        return combined_embeddings
