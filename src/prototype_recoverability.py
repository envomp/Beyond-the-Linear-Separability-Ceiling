import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def apply_unscaled_noise(embeds, noise_level):
    raw_noise = torch.randn_like(embeds)
    return ((1.0 - noise_level) * embeds) + (noise_level * raw_noise)


def run_numerical_tests(d, vocab_size=2 ** 20, trials=100):
    """ Empirically shows how distributed prototypes survive noise while single vectors don't. """
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    seq_lengths = [4096, 256, 16, 1]

    vocab = F.normalize(torch.randn(vocab_size, d), p=2, dim=-1)
    results = {nl: {k: 0 for k in seq_lengths} for nl in noise_levels}
    for nl in noise_levels:
        for _ in range(trials):
            target_idx = torch.randint(0, vocab_size, (1,)).item()
            target_concept = vocab[target_idx]

            for k in seq_lengths:
                base_tokens = target_concept.unsqueeze(0).expand(k, -1)
                noisy_tokens = apply_unscaled_noise(base_tokens, nl)
                prototype = noisy_tokens.mean(dim=0)
                prototype = F.normalize(prototype, p=2, dim=-1)
                proto_sims = prototype @ vocab.T
                if torch.argmax(proto_sims).item() == target_idx:
                    results[nl][k] += 1

    accuracy_dict = {nl: {k: results[nl][k] / trials for k in seq_lengths} for nl in noise_levels}
    return accuracy_dict, noise_levels, seq_lengths


def plot_results(all_results):
    from matplotlib.lines import Line2D

    plt.figure(figsize=(8, 5))
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    ax = plt.gca()
    dimensions = list(all_results.keys())
    linestyles = ['-', '--', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    first_dim = dimensions[0]
    _, noise_levels, seq_lengths = all_results[first_dim]
    for d_idx, d in enumerate(dimensions):
        accuracy_dict, _, _ = all_results[d]
        ls = linestyles[d_idx % len(linestyles)]
        for k_idx, k in enumerate(seq_lengths):
            y_vals = [accuracy_dict[nl][k] for nl in noise_levels]
            c = colors[k_idx % len(colors)]
            ax.plot(noise_levels, y_vals, marker='o', linestyle=ls, color=c, lw=2)

    ax.set_xlabel('Noise level ($\\alpha$)')
    ax.set_ylabel('Retrieval accuracy')
    ax.set_xticks(noise_levels)
    ax.grid(True, linestyle='--', alpha=0.7)

    color_handles = [
        Line2D([0], [0], color=colors[i % len(colors)], lw=2, marker='o', label=f'k = {k}')
        for i, k in enumerate(seq_lengths)
    ]
    legend_seq = ax.legend(handles=color_handles, title="Sequence length", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax.add_artist(legend_seq)
    style_handles = [Line2D([0], [0], color='black', linestyle=linestyles[i % len(linestyles)], lw=2, label=f'd = {d}') for i, d in enumerate(dimensions)]
    ax.legend(handles=style_handles, title="Dimension size", bbox_to_anchor=(1.02, 0.65), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("prototype_recoverability.pdf", format='pdf', bbox_inches='tight', pad_inches=0.13)


if __name__ == "__main__":
    # all_results = {}
    # dimensions = [16, 256, 4096]
    # print(f"Running tests for dimensions {dimensions}...")
    # for d in dimensions:
    #     acc_dict, noise_levels, seq_lengths = run_numerical_tests(d=d)
    #     all_results[d] = (acc_dict, noise_levels, seq_lengths)

    all_results = {16: (
        {0.0: {4096: 1.0, 256: 1.0, 16: 1.0, 1: 1.0}, 0.2: {4096: 1.0, 256: 1.0, 16: 1.0, 1: 0.02}, 0.4: {4096: 1.0, 256: 1.0, 16: 0.14, 1: 0.0}, 0.6: {4096: 1.0, 256: 0.92, 16: 0.0, 1: 0.0}, 0.8: {4096: 1.0, 256: 0.03, 16: 0.0, 1: 0.0},
         1.0: {4096: 0.0, 256: 0.0, 16: 0.0, 1: 0.0}}, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [4096, 256, 16, 1]), 256: (
        {0.0: {4096: 1.0, 256: 1.0, 16: 1.0, 1: 1.0}, 0.2: {4096: 1.0, 256: 1.0, 16: 1.0, 1: 0.15}, 0.4: {4096: 1.0, 256: 1.0, 16: 0.85, 1: 0.01}, 0.6: {4096: 1.0, 256: 1.0, 16: 0.01, 1: 0.0}, 0.8: {4096: 1.0, 256: 0.15, 16: 0.0, 1: 0.0},
         1.0: {4096: 0.0, 256: 0.0, 16: 0.0, 1: 0.0}}, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [4096, 256, 16, 1]), 4096: (
        {0.0: {4096: 1.0, 256: 1.0, 16: 1.0, 1: 1.0}, 0.2: {4096: 1.0, 256: 1.0, 16: 1.0, 1: 0.26}, 0.4: {4096: 1.0, 256: 1.0, 16: 0.8, 1: 0.0}, 0.6: {4096: 1.0, 256: 1.0, 16: 0.04, 1: 0.0}, 0.8: {4096: 1.0, 256: 0.21, 16: 0.0, 1: 0.0},
         1.0: {4096: 0.0, 256: 0.0, 16: 0.0, 1: 0.0}}, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [4096, 256, 16, 1])}
    print(all_results)
    plot_results(all_results)

# Semantically, a single vector is highly susceptible to noise where as prototype isn't.
# More interestingly, the resilience is mostly governed by sequence length than dimension size,
# allowing these distributed representations survive the low-dimensionality projections of attention heads.
