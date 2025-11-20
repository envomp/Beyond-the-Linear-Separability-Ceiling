import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrototypicalBatchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: [Batch_Size, Hidden_Dim] - Raw embeddings (will be normalized inside)
            labels: [Batch_Size] - Group IDs corresponding to each image
        """
        device = features.device

        features = F.normalize(features, p=2, dim=1)
        unique_labels = torch.unique(labels)
        contrast_losses = []
        for lbl in unique_labels:
            mask = (labels == lbl)
            current_group_features = features[mask]
            group_size = current_group_features.shape[0]
            if group_size < 2:
                continue

            positive_prototypes = []

            for i in range(group_size):
                other_indices = torch.arange(group_size, device=device) != i
                other_vectors = current_group_features[other_indices]
                centroid = torch.mean(other_vectors, dim=0)
                positive_prototypes.append(centroid)

            pos_prototypes_tensor = torch.stack(positive_prototypes)
            pos_prototypes_tensor = F.normalize(pos_prototypes_tensor, p=2, dim=1)
            pos_logits = (current_group_features * pos_prototypes_tensor).sum(dim=1).unsqueeze(1)

            neg_prototypes = []
            for other_lbl in unique_labels:
                if other_lbl == lbl:
                    continue
                other_mask = (labels == other_lbl)
                other_group_features = features[other_mask]
                other_centroid = torch.mean(other_group_features, dim=0)
                neg_prototypes.append(other_centroid)

            if len(neg_prototypes) > 0:
                neg_prototypes_tensor = torch.stack(neg_prototypes)
                neg_prototypes_tensor = F.normalize(neg_prototypes_tensor, p=2, dim=1)
                neg_logits = torch.matmul(current_group_features, neg_prototypes_tensor.T)
                all_logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
                targets = torch.zeros(all_logits.shape[0], dtype=torch.long, device=device)
                contrast_losses.append(F.cross_entropy(all_logits, targets))

        if len(contrast_losses) > 0:
            return torch.stack(contrast_losses).mean()
        else:
            return features.sum() * 0.0


def _calculate_single_item_loss(features, labels, item_index, temp):
    """Calculates the loss for a specific item index manually."""

    # 1. Normalize all inputs first
    norm_features = F.normalize(features, p=2, dim=1)

    target_vec = norm_features[item_index]
    target_label = labels[item_index]

    # 2. Calculate positive prototype (leave-one-out)
    # Find all other items in this group excluding current index
    mask_same_group = (labels == target_label)
    # We must select items that are in the same group BUT NOT the current item
    indices_in_group = torch.where(mask_same_group)[0]
    other_indices = indices_in_group[indices_in_group != item_index]

    if len(other_indices) == 0: return 0.0  # Should not happen in valid test case

    pos_vectors = norm_features[other_indices]
    pos_prototype = torch.mean(pos_vectors, dim=0)
    pos_prototype = F.normalize(pos_prototype, p=2, dim=0)

    # 3. Calculate negative prototypes
    unique_lbls = torch.unique(labels)
    neg_prototypes_list = []

    for lbl in unique_lbls:
        if lbl == target_label: continue
        neg_vectors = norm_features[labels == lbl]
        neg_centroid = torch.mean(neg_vectors, dim=0)
        neg_prototypes_list.append(neg_centroid)

    neg_prototypes = torch.stack(neg_prototypes_list)
    neg_prototypes = F.normalize(neg_prototypes, p=2, dim=1)

    # 4. Calculate logits & loss
    logit_pos = torch.dot(target_vec, pos_prototype) / temp
    logits_neg = torch.matmul(neg_prototypes, target_vec) / temp

    # Combine: [pos, neg1, neg2...]
    all_logits = torch.cat([logit_pos.unsqueeze(0), logits_neg])

    # Softmax cross entropy (target is always index 0)
    log_probs = F.log_softmax(all_logits, dim=0)
    loss = -log_probs[0]  # Negative log likelihood of the positive class

    return loss.item()


def run_tests():
    print("Running thorough prototypical loss tests...\n" + "=" * 40)
    torch.manual_seed(42)
    temp = 0.1
    loss_fn = PrototypicalBatchLoss(temperature=temp)

    # ==========================================
    # Manually calculate all
    # - avg of individual losses means we prioritize every image to be distanced from centroids of groups equally
    #   - standard approach is to distance images from images, but we distance images from concept centroids
    # - avg of group norms means we prioritize separation of group1 from rest equally much as group2 from rest
    #   - standard approach is to distance images from images, but we also distance concept centroids from concept centroids
    # - item 3 and 7 (big numbers) are ~equal because of symmetry
    # - group 2 items are (last four) ~equal because of symmetry
    # - group 1 and 2 mean losses are ~equal because of symmetry
    # - supports N groups for N-way decision boundary between classes
    # ==========================================
    print("\n[Test 0] Manually calculate all")

    # Setup: 2 Groups with different sizes and features
    # Group 0: 3 items
    # Group 1: 2 items
    features = torch.tensor([
        [1.0, 0.11],  # Item 0 (group 0)
        [1.0, 0.12],  # Item 1 (group 0)
        [1.0, 0.13],  # Item 2 (group 0)
        [0.0, 0.12],  # Item 3 (group 0)
        [0.11, 1.0],  # Item 4 (group 1)
        [0.12, 1.0],  # Item 5 (group 1)
        [0.13, 1.0],  # Item 6 (group 1)
        [0.12, 0.0],  # Item 7 (group 1)
        [1.0, 1.00],  # Item 8 (group 2)
        [0.99, 1.0],  # Item 9 (group 2)
        [0.99, 0.99],  # Item 10 (group 2)
        [0.99, 0.99],  # Item 11 (group 2)
    ], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    # 1. Get actual loss
    actual_loss = loss_fn(features, labels)

    # 2. Calculate manual losses
    individual_losses = []
    losses_by_group = {}
    for i in range(len(features)):
        lbl = labels[i].item()
        loss_val = _calculate_single_item_loss(features, labels, i, temp)
        individual_losses.append(loss_val)
        if lbl not in losses_by_group:
            losses_by_group[lbl] = []
        losses_by_group[lbl].append(loss_val)

    # 3. Aggregate manual losses (mean of group means)
    group_means = []
    for lbl in sorted(losses_by_group.keys()):
        g_mean = sum(losses_by_group[lbl]) / len(losses_by_group[lbl])
        group_means.append(g_mean)

    # Final expected loss is the average of the group means
    total_expected_loss = sum(group_means) / len(group_means)

    print(f"\nIndividual losses: {[f'{x:.4f}' for x in individual_losses]}")
    print(f"Group means:       {[f'{x:.4f}' for x in group_means]}")
    print(f"-" * 30)
    print(f"Total expected:    {total_expected_loss:.6f}")
    print(f"Actual output:     {actual_loss.item():.6f}")

    assert math.isclose(actual_loss.item(), total_expected_loss, rel_tol=1e-5)
    print("\n> PASSED: Manual aggregation matches Class output.")

    # ====================================================
    # Test 1: Loss is symmetrical (loss for each element is identical when distances are identical)
    # ====================================================
    print("\n[Test 1] Loss is symmetrical")
    # Scenario:
    # Group 0: A=[1, 0], B=[0, 1]  (orthogonal, should punish)
    # Group 1: C=[0, -1], D=[-1, 0] (opposite to group 0)

    features = torch.tensor([
        [1.0, 0.0],  # Index 0 (group 0)
        [0.0, 1.0],  # Index 1 (group 0)
        [0.0, -1.0],  # Index 2 (group 1)
        [-1.0, 0.0]  # Index 3 (group 1)
    ], requires_grad=True)
    group_ids = torch.tensor([0, 0, 1, 1])

    # Positive prototype (leave-one-out): mean of group 0 excluding index 0.
    #    -> Only index 1 remains: [0, 1]. normalized: [0, 1].
    pos_proto_0 = torch.tensor([0.0, 1.0])

    # Negative prototype: mean of group 1 (indices 2, 3).
    #    -> mean([0, -1], [-1, 0]) = [-0.5, -0.5].
    #    -> normalized: [-0.7071, -0.7071]
    neg_vec = torch.tensor([-0.5, -0.5])
    neg_proto_0 = neg_vec / neg_vec.norm(p=2)

    # 3. Logits for index 0
    feat_0 = features[0]  # [1.0, 0.0]
    dot_pos = torch.dot(feat_0, pos_proto_0)  # Should be 0.0
    dot_neg = torch.dot(feat_0, neg_proto_0)  # Should be -0.707106...
    logit_pos = dot_pos / temp
    logit_neg = dot_neg / temp

    # 4. Expected loss for item 0 (cross entropy with target 0)
    #    Formula: -log( exp(pos) / (exp(pos) + exp(neg)) )
    #    Using logsumexp for numerical stability in manual calc
    max_val = max(logit_pos, logit_neg)
    log_sum_exp = max_val + math.log(math.exp(logit_pos - max_val) + math.exp(logit_neg - max_val))
    expected_loss_item_0 = log_sum_exp - logit_pos

    actual_loss = loss_fn(features, group_ids)

    print(f"  > Manual calc loss for item 0: {expected_loss_item_0:.6f}")
    print(f"  > Actual class output:         {actual_loss.item():.6f}")
    assert torch.isclose(actual_loss, torch.tensor(expected_loss_item_0), atol=1e-5), "FAILED: The code result does not match the manual math."
    print("  > PASSED: Exact math match.")

    # ====================================================
    # Test 2: Large scale & variable sizes
    # ====================================================
    print("\n[Test 2] Large scale & variable sizes")
    # Scenario:
    # Group A: 10 items
    # Group B: 2 items (minimum required)
    # Group D: 1 item (valid negative, invalid anchor)
    torch.manual_seed(42)
    feats_A = torch.randn(10, 128)
    feats_B = torch.randn(2, 128)
    feats_D = torch.randn(1, 128)
    features_large = torch.cat([feats_A, feats_B, feats_D], dim=0)
    features_large.requires_grad = True
    group_ids = torch.cat([torch.zeros(10), torch.ones(2), torch.ones(1) * 2]).long()
    loss_large = loss_fn(features_large, group_ids)
    loss_large.backward()

    print(f"  > Large batch loss: {loss_large.item():.4f}")
    assert not torch.isnan(loss_large), "FAILED: Large batch produced NaN"
    assert features_large.grad is not None, "FAILED: No gradients flowing"
    grad_norm_last = features_large.grad[-1].norm().item()
    print(f"  > Gradient on single item (group D): {grad_norm_last:.6f}")
    assert grad_norm_last > 0.0, "FAILED: Single-item group acted as a negative but received no gradient."
    print("  > PASSED: Large scale and variable sizes handled correctly.")

    # ====================================================
    # Test 3: Normalization robustness (input vectors with huge magnitude should give same loss as normalized ones)
    # ====================================================
    print("\n[Test 3] Normalization robustness")
    f_norm = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    group_ids = torch.tensor([0, 0, 1, 1])
    f_huge = f_norm * 1000.0
    l_norm = loss_fn(f_norm, group_ids)
    l_huge = loss_fn(f_huge, group_ids)
    print(f"  > Normal input loss: {l_norm.item():.6f}")
    print(f"  > Huge input loss:   {l_huge.item():.6f}")
    assert torch.isclose(l_norm, l_huge, atol=1e-5), "FAILED: Loss changed significantly with input magnitude."
    print("  > PASSED: Input normalization is working.")

    # ====================================================
    # Test 4: Duplicate vectors (crash test)
    # ====================================================
    print("\n[Test 4] Duplicate vectors")
    f_ident = torch.ones((6, 4), requires_grad=True)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    loss_ident = loss_fn(f_ident, group_ids)
    print(f"  > Identical vectors loss: {loss_ident.item():.4f}")
    assert not torch.isnan(loss_ident), "FAILED: Identical vectors caused NaN."
    print("  > PASSED: Handled identical inputs.")


if __name__ == "__main__":
    run_tests()
