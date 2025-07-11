from torch.utils.data import default_collate
import torch
from typing import Literal


def next_word_prediction_labels(input_ids, answer_ids, strategy: Literal["answer_only", "all", "masked"] = "answer_only"):
    """
    > strategy = "answer_only", input_ids = i n p u t Q, answer_ids = V
        i n p u t Q
        - - - - - V

    > strategy = "all", input_ids = i n p u t Q, answer_ids = V
        i n p u t Q
        n p u t Q V

    > strategy = "masked", input_ids = i n p u t Q1 Q2 Q3, answer_ids = V1 V2 V3
        i n p u t Q1 Q2 Q3
        - - - - - V1 V2 V3
    """
    if strategy == "answer_only":
        input = input_ids + answer_ids
        labels = [-100 for _ in input_ids] + answer_ids
        input.pop(-1)
        labels.pop(0)
        return labels, input

    if strategy == "all":
        input = input_ids + answer_ids
        labels = input_ids + answer_ids
        input.pop(-1)
        labels.pop(0)
        return labels, input

    if strategy == "masked":
        input = input_ids
        labels = [-100 for _ in range(len(input_ids) - len(answer_ids))] + answer_ids
        return labels, input

    raise ValueError(f"unknown strategy: {strategy}")


def pad_collate(batch, padding, length=None):
    max_size = max([len(x["input_ids"]) for x in batch]) if not length else length
    collated = []

    for elem in batch:
        copy = {}
        copy["input_ids"] = torch.tensor(elem["input_ids"] + [padding] * (max_size - len(elem["input_ids"])))

        copy["labels"] = torch.tensor(elem["labels"] + [-100] * (max_size - len(elem["labels"])))

        if elem.get("global_attention_mask") is not None:
            copy["global_attention_mask"] = torch.tensor(elem["global_attention_mask"] + [False] * (max_size - len(elem["global_attention_mask"])))

        if elem.get("type_embeddings") is not None:
            copy["type_embeddings"] = torch.tensor(elem["type_embeddings"] + [[0, 0]] * (max_size - len(elem["type_embeddings"])))

        if elem.get("depth") is not None:
            copy["depth"] = elem["depth"]

        collated.append(copy)

    return default_collate(collated)


def labels_to_one_hot(labels, num_classes):
    batch_size, seq_len = labels.shape
    one_hot = torch.zeros(batch_size, seq_len, num_classes, dtype=labels.dtype, device=labels.device)
    valid_indices = labels != -100
    one_hot[valid_indices, labels[valid_indices]] = 1
    return one_hot


if __name__ == '__main__':
    labels, input_seq = next_word_prediction_labels([1, 2, 3], [4, 5], "answer_only")
    assert labels == [-100, -100, 4, 5]
    assert input_seq == [1, 2, 3, 4]

    labels, input_seq = next_word_prediction_labels([1, 2, 3], [4, 5], "all")
    assert labels == [2, 3, 4, 5]
    assert input_seq == [1, 2, 3, 4]

    labels, input_seq = next_word_prediction_labels([1, 2, 3, 4, 5, 6], [7, 8, 9], "masked")
    assert labels == [-100, -100, -100, 7, 8, 9]
    assert input_seq == [1, 2, 3, 4, 5, 6]
