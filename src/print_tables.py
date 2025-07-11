from scipy.stats import chi2_contingency, norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

P_VALUE = 0.05


def extract_correctness(filename):
    correctness_list = []
    with open(filename, 'r') as file:
        for line in file:
            if 'SKIP' in line or "got='None'" in line:
                correctness_list.append(None)
            elif ', correct:' in line:
                correctness_value = line.split(', correct:')[1].strip()
                correctness_list.append('True' in correctness_value)
            elif '| equals:' in line:
                correctness_value = line.split('| equals:')[1].strip()
                correctness_list.append('True' in correctness_value)
            elif '| expected:' in line:
                correctness_value = "expected:'cat_2' | got='cat_2" in line or "expected:'cat_1' | got='cat_1" in line
                correctness_list.append(correctness_value)

    return correctness_list


def accuracy(vals):
    non_skip = [x for x in vals if x is not None]
    return round(sum(non_skip) / len(non_skip) * 100, 1)


def accuracy_with_error_margin(vals, bound=None):
    non_skip = [x for x in vals if x is not None]
    n = len(non_skip)
    vals_arr = np.array(non_skip)
    p = np.mean(vals_arr)
    standard_error = np.sqrt((p * (1 - p)) / n)

    alpha = P_VALUE
    z_score = norm.ppf(1 - alpha / 2)

    margin_of_error_proportion = z_score * standard_error
    accuracy_percentage = p * 100
    margin_of_error_percentage = margin_of_error_proportion * 100
    if not bound:
        return f"{accuracy_percentage:.1f} \\pm {margin_of_error_percentage:.1f}"
    elif bound == "upper":
        return p + margin_of_error_proportion
    elif bound == "lower":
        return p - margin_of_error_proportion
    else:
        return None


def agreement(correctness_values_1, correctness_values_2):
    tt_obs = 0  # Observed: M1=True, M2=True
    tf_obs = 0  # Observed: M1=True, M2=False
    ft_obs = 0  # Observed: M1=False, M2=True
    ff_obs = 0  # Observed: M1=False, M2=False
    processed = 0

    for i in range(len(correctness_values_1)):
        m1 = correctness_values_1[i]
        m2 = correctness_values_2[i]
        if m1 is None or m2 is None:
            continue
        else:
            processed += 1

        if m1 and m2:
            tt_obs += 1
        elif m1 and not m2:
            tf_obs += 1
        elif not m1 and m2:
            ft_obs += 1
        elif not m1 and not m2:
            ff_obs += 1

    observed_counts = np.array([[tt_obs, tf_obs], [ft_obs, ff_obs]])
    chi2, p_value, dof, expected_counts = chi2_contingency(observed_counts)
    observed_agreement, expected_agreement = tt_obs + ff_obs, expected_counts[0, 0] + expected_counts[1, 1]
    observed_agreement = round(observed_agreement / processed * 100, 1)
    expected_agreement = round(float(expected_agreement / processed * 100), 1)
    return observed_agreement, expected_agreement, p_value


def agreement_statement(cor_vals_1, cor_vals_2):
    observed, expected, p_value = agreement(cor_vals_1, cor_vals_2)
    is_significant = p_value < P_VALUE
    statement = f"observed={observed}, expected={expected}, significant={is_significant}, p_value={p_value}"
    return statement


def imbalances_PEFT():
    models = ["phi", "pixtral", "gemma3_4b"]
    datasets = ["openworld", "hoi"]
    pefts = ["lora", "sim_lora"]

    sim_runs = ["{model}_{dataset}_{peft}_similarity_vision.txt"]
    gen_runs = ["{model}_{dataset}_{peft}_eval.txt"]
    filters = [("pos", lambda x: x[0::2]), ("neg", lambda x: x[1::2]), ("avg", lambda x: x[0::])]

    accuracies = {}
    for model in models:
        model_accuracies = {}
        accuracies[model] = model_accuracies

        for dataset in datasets:
            gen_accuracies = [[] for _ in range(len(pefts) + 1)]
            sim_accuracies = [[] for _ in range(len(pefts) + 1)]
            model_accuracies[dataset] = (gen_accuracies, sim_accuracies)

            print(model, dataset)
            eval_direct = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/baselines/results_{dataset}_direct_interleaved.txt")
            sim_final = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/similarity_baselines/results_{dataset}_vision_interleaved.txt")
            for filter_name, filter_f in filters:
                gen_accuracies[0].append(accuracy(filter_f(eval_direct)))
                sim_accuracies[0].append(accuracy(filter_f(sim_final)))

            for filter_name, filter_f in filters:
                for i, peft in enumerate(pefts):
                    for file in sim_runs:
                        path = f'../result/PEFT/{file.replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}'
                        cor_vals = filter_f(extract_correctness(path))
                        print(path, filter_name, accuracy(cor_vals))
                        sim_accuracies[1 + i].append(accuracy(cor_vals))
                    for file in gen_runs:
                        path = f'../result/PEFT/{file.replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}'
                        cor_vals = filter_f(extract_correctness(path))
                        print(path, filter_name, accuracy(cor_vals))
                        gen_accuracies[1 + i].append(accuracy(cor_vals))
    return accuracies


def split(items, items_per_split=200, splits=4):
    res = []
    for i in range(splits):
        res.append(items[i * items_per_split:(i + 1) * items_per_split])
    return res


def splits_PEFT(items_per_split=200, splits=4):
    models = ["phi", "pixtral", "gemma3_4b"]
    datasets = ["hoi"]
    pefts = ["full", "sim_full", "lora", "sim_lora"]

    sim_runs = ["{model}_{dataset}_{peft}_similarity_final.txt"]
    gen_runs = ["{model}_{dataset}_{peft}_eval.txt"]

    accuracies = {}
    for model in models:
        model_accuracies = {}
        accuracies[model] = model_accuracies

        for dataset in datasets:
            gen_accuracies = [[] for _ in range(len(pefts) + 1)]
            sim_accuracies = [[] for _ in range(len(pefts) + 1)]
            model_accuracies[dataset] = (gen_accuracies, sim_accuracies)

            print(model, dataset)
            eval_direct = split(extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/baselines/results_{dataset}_direct_interleaved.txt"), items_per_split, splits)
            sim_final = split(extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/similarity_baselines/results_{dataset}_final_interleaved.txt"), items_per_split, splits)
            gen_accuracies[0].append([accuracy(x) for x in eval_direct])
            sim_accuracies[0].append([accuracy(x) for x in sim_final])

            for i, peft in enumerate(pefts):
                for file in sim_runs:
                    path = f'../result/PEFT/{file.replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}'
                    cor_vals = split(extract_correctness(path), items_per_split, splits)
                    sim_accuracies[1 + i].append([accuracy(x) for x in cor_vals])
                for file in gen_runs:
                    path = f'../result/PEFT/{file.replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}'
                    cor_vals = split(extract_correctness(path), items_per_split, splits)
                    gen_accuracies[1 + i].append([accuracy(x) for x in cor_vals])
    return accuracies


def accuracies_PEFT(highlight=True, brief=False, cross_domain=False, phi_ablation=False, no_vision=False):
    if not phi_ablation:
        models = ["phi", "pixtral", "gemma3_4b"]
        datasets = ["openworld", "hoi"]
        if brief:
            pefts = ["lora", "sim_lora"]
            pretty_names = ["LoRA ($\\mathcal{L}_{\\text{NT}}$)",
                            "LoRA ($\\mathcal{L}_{\\text{combined}}$)"]
        else:
            pefts = ["postfix", "full", "sim_full", "lora", "sim_lora"]
            pretty_names = ["Postfix tuning ($\\mathcal{L}_{\\text{NT}}$)",
                            "Prompt tuning ($\\mathcal{L}_{\\text{NT}}$)",
                            "Prompt tuning ($\\mathcal{L}_{\\text{combined}}$)",
                            "LoRA ($\\mathcal{L}_{\\text{NT}}$)",
                            "LoRA ($\\mathcal{L}_{\\text{combined}}$)"]
    else:
        models = ["phi"]
        datasets = ["openworld"]
        pefts = ["vision", "connector", "postfix", "full", "lora", "lora_no_vision"]
        pretty_names = ["Vision bias tuning ($\\mathcal{L}_{\\text{NT}}$)",
                        "Connector tuning ($\\mathcal{L}_{\\text{NT}}$)",
                        "Postfix tuning ($\\mathcal{L}_{\\text{NT}}$)",
                        "Prompt tuning ($\\mathcal{L}_{\\text{NT}}$)",
                        "LoRA ($\\mathcal{L}_{\\text{NT}}$)",
                        "LoRA ($\\mathcal{L}_{\\text{NT}}$ no-vision)"]

    if cross_domain:  # domain generalization
        sim_runs = ["{model}_{dataset}_{peft}_similarity_vision.txt", "{model}_{dataset}_{peft}_similarity_final.txt", "{model}_cross_domain_{dataset}_{peft}_similarity_final.txt"]
        gen_runs = ["{model}_{dataset}_{peft}_eval.txt", "{model}_cross_domain_{dataset}_{peft}_eval.txt"]
    elif no_vision:
        sim_runs = []
        gen_runs = ["{model}_{dataset}_{peft}_eval.txt", "{model}_{dataset}_{peft}_no_vision_eval.txt"]
    else:  # syntactic generalization
        sim_runs = ["{model}_{dataset}_{peft}_similarity_vision.txt", "{model}_{dataset}_{peft}_similarity_final.txt", "{model}_{dataset}_{peft}_similarity_final_labeled_prompt.txt"]
        gen_runs = ["{model}_{dataset}_{peft}_eval.txt", "{model}_{dataset}_{peft}_eval_labeled_prompt.txt"]

    accuracies = {}
    for model in models:
        model_accuracies = {}
        accuracies[model] = model_accuracies

        for dataset in datasets:
            gen_accuracies = [[] for _ in range(len(pefts) + 1)]
            sim_accuracies = [[] for _ in range(len(pefts) + 1)]
            model_accuracies[dataset] = (gen_accuracies, sim_accuracies)

            print(model, dataset)
            eval_direct = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/baselines/results_{dataset}_direct_interleaved.txt")
            eval_direct_labeled = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/baselines/results_{dataset}_direct_labeled.txt")
            sim_vision = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/similarity_baselines/results_{dataset}_vision_interleaved.txt")
            sim_final = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/similarity_baselines/results_{dataset}_final_interleaved.txt")
            sim_final_labeled = extract_correctness(f"../result/{model.replace("gemma3", "gemma")}/similarity_baselines/results_{dataset}_final_labeled.txt")
            gen_surpassed = accuracy_with_error_margin(eval_direct, bound="lower") > accuracy_with_error_margin(sim_vision, bound="upper")
            gen_crossing = accuracy_with_error_margin(eval_direct, bound="upper") > accuracy_with_error_margin(sim_vision, bound="lower")
            gen_labeled_surpassed = accuracy_with_error_margin(eval_direct_labeled, bound="lower") > accuracy_with_error_margin(sim_vision, bound="upper")
            gen_labeled_crossing = accuracy_with_error_margin(eval_direct_labeled, bound="upper") > accuracy_with_error_margin(sim_vision, bound="lower")

            gen_accuracies[0].append(accuracy_with_error_margin(eval_direct))
            if gen_crossing and highlight:
                gen_accuracies[0][-1] = f"\\mathbf{{{gen_accuracies[0][-1]}}}"
            if gen_surpassed and highlight:
                gen_accuracies[0][-1] = f"\\underline{{{gen_accuracies[0][-1]}}}"
            gen_accuracies[0].append(accuracy_with_error_margin(eval_direct_labeled))
            if gen_labeled_crossing and highlight:
                gen_accuracies[0][-1] = f"\\mathbf{{{gen_accuracies[0][-1]}}}"
            if gen_labeled_surpassed and highlight:
                gen_accuracies[0][-1] = f"\\underline{{{gen_accuracies[0][-1]}}}"

            vals_out = []
            for compare_method, compare_to in [("I", eval_direct), ("O", eval_direct_labeled)]:
                observed, expected, p_value = agreement(sim_vision, compare_to)
                if p_value < P_VALUE:
                    vals_out.append(("" if observed > expected else "-") + compare_method)
            sim_accuracies[0].append(accuracy_with_error_margin(sim_vision) + "^{" + ",".join(vals_out) + "}")

            vals_out = []
            for compare_method, compare_to in [("I", eval_direct)]:
                observed, expected, p_value = agreement(sim_final, compare_to)
                if p_value < P_VALUE:
                    vals_out.append(("" if observed > expected else "-") + compare_method)
            sim_accuracies[0].append(accuracy_with_error_margin(sim_final) + "^{" + ",".join(vals_out) + "}")

            vals_out = []
            for compare_method, compare_to in [("O", eval_direct_labeled)]:
                observed, expected, p_value = agreement(sim_final_labeled, compare_to)
                if p_value < P_VALUE:
                    vals_out.append(("" if observed > expected else "-") + compare_method)
            sim_accuracies[0].append(accuracy_with_error_margin(sim_final_labeled) + "^{" + ",".join(vals_out) + "}")

            for i, peft in enumerate(pefts):
                if sim_runs:
                    cor_vals_vision = extract_correctness(f'../result/PEFT/{sim_runs[0].replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}')
                    cor_vals_final = extract_correctness(f'../result/PEFT/{sim_runs[1].replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}')
                    cor_vals_final_labeled = extract_correctness(f'../result/PEFT/{sim_runs[2].replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}')
                cor_vals_gen = extract_correctness(f'../result/PEFT/{gen_runs[0].replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}')
                cor_vals_gen_labeled = extract_correctness(f'../result/PEFT/{gen_runs[1].replace("{model}", model).replace("{dataset}", dataset).replace("{peft}", peft)}')

                if sim_runs:
                    vals_out = []
                    for compare_method, compare_to in [("I", cor_vals_gen), ("O", cor_vals_gen_labeled)]:
                        observed, expected, p_value = agreement(cor_vals_vision, compare_to)
                        if p_value < P_VALUE:
                            vals_out.append(("" if observed > expected else "-") + compare_method)
                    sim_accuracies[1 + i].append(accuracy_with_error_margin(cor_vals_vision) + "^{" + ",".join(vals_out) + "}")

                    vals_out = []
                    for compare_method, compare_to in [("I", cor_vals_gen)]:
                        observed, expected, p_value = agreement(cor_vals_final, compare_to)
                        if p_value < P_VALUE:
                            vals_out.append(("" if observed > expected else "-") + compare_method)
                    sim_accuracies[1 + i].append(accuracy_with_error_margin(cor_vals_final) + "^{" + ",".join(vals_out) + "}")

                    vals_out = []
                    for compare_method, compare_to in [("O", cor_vals_gen_labeled)]:
                        observed, expected, p_value = agreement(cor_vals_final_labeled, compare_to)
                        if p_value < P_VALUE:
                            vals_out.append(("" if observed > expected else "-") + compare_method)
                    sim_accuracies[1 + i].append(accuracy_with_error_margin(cor_vals_final_labeled) + "^{" + ",".join(vals_out) + "}")
                else:
                    sim_accuracies[1 + i].append("-")
                    sim_accuracies[1 + i].append("-")
                    sim_accuracies[1 + i].append("-")

                gen_accuracies[1 + i].append(accuracy_with_error_margin(cor_vals_gen))
                if highlight and accuracy_with_error_margin(cor_vals_gen, bound="upper") > accuracy_with_error_margin(cor_vals_vision, bound="lower"):
                    gen_accuracies[1 + i][-1] = f"\\mathbf{{{gen_accuracies[1 + i][-1]}}}"
                if highlight and accuracy_with_error_margin(cor_vals_gen, bound="lower") > accuracy_with_error_margin(cor_vals_vision, bound="upper"):
                    gen_accuracies[1 + i][-1] = f"\\underline{{{gen_accuracies[1 + i][-1]}}}"

                gen_accuracies[1 + i].append(accuracy_with_error_margin(cor_vals_gen_labeled))
                if highlight and accuracy_with_error_margin(cor_vals_gen_labeled, bound="upper") > accuracy_with_error_margin(cor_vals_vision, bound="lower"):
                    gen_accuracies[1 + i][-1] = f"\\mathbf{{{gen_accuracies[1 + i][-1]}}}"
                if highlight and accuracy_with_error_margin(cor_vals_gen_labeled, bound="lower") > accuracy_with_error_margin(cor_vals_vision, bound="upper"):
                    gen_accuracies[1 + i][-1] = f"\\underline{{{gen_accuracies[1 + i][-1]}}}"
    return accuracies, pretty_names


def agreement_baselines():
    models = ["phi", "gemma_4b", "gemma_27b", "intern_14b", "intern_78b",  "qwen_7b", "qwen_72b", "pixtral"]
    datasets = ["openworld", "hoi"]
    methods = ["interleaved", "interleaved_test_first", "labeled", "labeled_test_first"]
    isolation_methods = ["batched", "single"]

    collected_rows = []
    ceiling_strength = {"crossed": [],
                        "surpassed": {"pathway": [], "repr": []},
                        "all": {"sim_vision": [], "sim_final": []},
                        "inversed": {"sim_vision": [], "sim_final": []}}

    for model in models:
        for dataset in datasets:
            for method in methods:
                eval_direct = extract_correctness(f"../result/{model}/baselines/results_{dataset}_direct_{method}.txt")
                eval_cot = extract_correctness(f"../result/{model}/baselines/results_{dataset}_cot_{method}.txt")
                sim_vision = extract_correctness(f"../result/{model}/similarity_baselines/results_{dataset}_vision_{method}.txt")
                sim_final = extract_correctness(f"../result/{model}/similarity_baselines/results_{dataset}_final_{method}.txt")

                print(f"{model} {dataset} {method}")
                print("vision -> direct", agreement_statement(sim_vision, eval_direct))
                print("vision -> cot", agreement_statement(sim_vision, eval_cot))
                print("final -> direct", agreement_statement(sim_final, eval_direct))
                print("final -> cot", agreement_statement(sim_final, eval_cot))
                cor_vision = []
                cor_final = []
                for vals_in, name, vals_out in [(sim_vision, "sim_vision", cor_vision), (sim_final, "sim_final", cor_final)]:
                    for compare_method, compare_to in [("D", eval_direct), ("C", eval_cot)]:
                        observed, expected, p_value = agreement(vals_in, compare_to)
                        if p_value < P_VALUE:
                            ceiling_strength["all"][name].append(f"{model}_{dataset}_{method}")
                            if observed < expected:
                                ceiling_strength["inversed"][name].append(f"{model}_{dataset}_{method}")
                            vals_out.append(("" if observed > expected else "-") + compare_method)

                for generative_eval, gen_name in [(eval_direct, "direct"), (eval_cot, "cot")]:
                    if accuracy_with_error_margin(eval_cot, bound="upper") > accuracy_with_error_margin(sim_vision, bound="lower"):
                        ceiling_strength["crossed"].append(f"{model}_{dataset}_{method}_{gen_name}")
                    if accuracy_with_error_margin(eval_cot, bound="lower") > accuracy_with_error_margin(sim_vision, bound="upper"):
                        if accuracy_with_error_margin(eval_cot, bound="lower") < accuracy_with_error_margin(sim_final, bound="upper"):
                            ceiling_strength["surpassed"]["repr"].append(f"{model}_{dataset}_{method}_{gen_name}")
                        else:
                            ceiling_strength["surpassed"]["pathway"].append(f"{model}_{dataset}_{method}_{gen_name}")

                collected_rows.append({
                    'model': model,
                    'dataset': dataset,
                    'method': method,
                    'gen_direct': accuracy_with_error_margin(eval_direct),
                    'gen_cot': accuracy_with_error_margin(eval_cot),
                    'sim_vision': accuracy_with_error_margin(sim_vision),
                    'sim_final': accuracy_with_error_margin(sim_final),
                    'cor_vision': cor_vision,
                    'cor_final': cor_final,
                })

            for method in isolation_methods:
                sim_vision = extract_correctness(f"../result/{model}/similarity_baselines/results_{dataset}_isolation_vision_{method}.txt")
                sim_final = extract_correctness(f"../result/{model}/similarity_baselines/results_{dataset}_isolation_final_{method}.txt")
                collected_rows.append({
                    'model': model,
                    'dataset': dataset,
                    'method': method,
                    'gen_direct': "",
                    'gen_cot': "",
                    'sim_vision': accuracy_with_error_margin(sim_vision),
                    'sim_final': accuracy_with_error_margin(sim_final),
                    'cor_vision': [],
                    'cor_final': [],
                })
    return collected_rows, ceiling_strength


def baseline_table_latex(rows):
    latex_parts = []

    actual_caption = "Separability ceiling analysis"
    actual_label = "tab:combined_similarity_performance"

    latex_parts.append("\\begin{table}[htbp]")
    latex_parts.append("    \\centering")
    latex_parts.append(f"    \\caption{{{actual_caption}}}")
    latex_parts.append(f"    \\label{{{actual_label}}}")
    latex_parts.append("    \\resizebox{\\textwidth}{!}{")
    latex_parts.append("        \\begin{tabular}{l l l c c c c}")
    latex_parts.append("            \\toprule")
    latex_parts.append("            Model & Dataset   & Method / Prompt Strategy     & Direct acc (\\%) & CoT acc (\\%) & Sim. acc (vision, \\%) & Sim. acc (final, \\%) \\\\")
    latex_parts.append("            \\midrule")

    data_for_table = {}
    for row in rows:
        model = row['model']
        dataset = row['dataset']
        method = row['method']
        if model not in data_for_table:
            data_for_table[model] = {}
        if dataset not in data_for_table[model]:
            data_for_table[model][dataset] = {}
        data_for_table[model][dataset][method] = row

    model_display_names = {"phi": "Phi", "pixtral": "Pixtral",
                           "gemma_4b": "Gemma3 4B", "gemma_27b": "Gemma3 27B",
                           "intern_14b": "InternVL3 14B", "intern_78b": "InternVL3 78B",
                           "qwen_7b": "Qwen2.5-VL 7B", "qwen_72b": "Qwen2.5-VL 72B"}
    dataset_display_names = {"openworld": "OpenWorld", "hoi": "HOI"}

    method_display_map = {
        "interleaved": "Prompt context (Interleaved)",
        "labeled": "Prompt context (Labeled)",
        "interleaved_test_first": "Prompt context (Interleaved test first)",
        "labeled_test_first": "Prompt context (Labeled test first)",
        "single": "Single context",
        "batched": "Batched context"
    }

    prompts = ['interleaved', 'interleaved_test_first', 'labeled', 'labeled_test_first', 'single', 'batched']
    models = ["phi", "pixtral", "gemma_4b", "gemma_27b", "intern_14b", "intern_78b", "qwen_7b", "qwen_72b"]
    datasets = ["openworld", "hoi"]

    num_models_in_config = len(models)
    for model_idx, model_key in enumerate(models):
        model_data_from_rows = data_for_table[model_key]
        latex_parts.append("            " + model_display_names[model_key])

        for dataset_idx, dataset_key in enumerate(datasets):
            current_dataset_methods_data = model_data_from_rows[dataset_key]
            for method_idx, method_key_from_config in enumerate(prompts):
                row_data_for_method = current_dataset_methods_data.get(method_key_from_config)
                dataset_cell_str = dataset_display_names[dataset_key] if method_idx == 0 else ""
                method_display_name_str = method_display_map[method_key_from_config]

                val_direct = "$" + row_data_for_method.get('gen_direct') + "$"
                val_cot = "$" + row_data_for_method.get('gen_cot') + "$"
                val_sim_vision = row_data_for_method.get('sim_vision')
                sim_vision_str = "$" + val_sim_vision + "^{" + ",".join(row_data_for_method.get('cor_vision')) + "}" + "$"
                val_sim_final = row_data_for_method.get('sim_final')
                sim_final_str = "$" + val_sim_final + "^{" + ",".join(row_data_for_method.get('cor_final')) + "}" + "$"
                line = f"            & {dataset_cell_str} & {method_display_name_str} & {val_direct} & {val_cot} & {sim_vision_str} & {sim_final_str} \\\\"
                latex_parts.append(line)

            if dataset_idx < len(datasets) - 1:
                latex_parts.append("            \\cmidrule(lr){2-7}")

        if model_idx < num_models_in_config - 1:
            latex_parts.append("            \\midrule")

    latex_parts.append("            \\bottomrule")
    latex_parts.append("        \\end{tabular}")
    latex_parts.append("    }")
    latex_parts.append("\\end{table}")

    return "\n".join(latex_parts)


def class_imbalance_table_latex(accuracies_rows_data):
    model_keys = ["phi", "pixtral", "gemma3_4b"]
    dataset_keys = ["openworld", "hoi"]

    method_latex_names = [
        "Direct (interleaved)",
        r"LoRA ($\mathcal{L}_{\text{NT}}$)",
        r"LoRA ($\mathcal{L}_{\text{combined}}$)"
    ]

    num_datasets = len(dataset_keys)

    latex_column_headers = r"""            Model
            & Dataset
            & Method
            & \begin{tabular}[c]{@{}c@{}}
                  gen. acc. \\ pos. (\%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  gen. acc. \\ neg. (\%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  gen. acc. \\ avg. (\%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  LSC \\ (pos., \%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  LSC \\ (neg., \%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  LSC \\ (avg., \%)
            \end{tabular} \\"""

    table_parts = []

    table_caption = "Class imbalances across datasets and models"
    table_label = "tab:lora_classes_split"

    table_parts.append(r"% Combined table for all datasets and models")
    table_parts.append(r"\begin{table*}[htbp]")
    table_parts.append(r"    \centering")
    table_parts.append(f"    \\caption{{{table_caption}}}")
    table_parts.append(f"    \\label{{{table_label}}}")
    table_parts.append(r"    \resizebox{\textwidth}{!}{%")
    table_parts.append(r"        \begin{tabular}{@{}l l l c c c c c c@{}}")
    table_parts.append(r"            \toprule")
    table_parts.append(latex_column_headers)
    table_parts.append(r"            \midrule")

    for model_idx, model_key in enumerate(model_keys):
        model_name_pretty = model_key.capitalize()

        if model_key not in accuracies_rows_data:
            print(f"Warning: Data for model '{model_key}' not found. Skipping.")
            continue

        for dataset_idx, dataset_key in enumerate(dataset_keys):
            dataset_name_pretty = dataset_key.capitalize()

            if dataset_key not in accuracies_rows_data[model_key]:
                print(f"Warning: Data for dataset '{dataset_key}' under model '{model_key}' not found. Skipping.")
                continue

            gen_accuracies, sim_accuracies = accuracies_rows_data[model_key][dataset_key]

            for method_idx, method_name in enumerate(method_latex_names):
                g_vals = gen_accuracies[method_idx]
                s_vals = sim_accuracies[method_idx]

                data_cells_str = (
                    rf"{g_vals[0]:.1f} & {g_vals[1]:.1f} & {g_vals[2]:.1f} & "
                    rf"{s_vals[0]:.1f} & {s_vals[1]:.1f} & {s_vals[2]:.1f}"
                )

                current_row_parts = ["            "]

                if dataset_idx == 0 and method_idx == 0:
                    current_row_parts.append(model_name_pretty)
                current_row_parts.append(" & ")

                if method_idx == 0:
                    current_row_parts.append(dataset_name_pretty)
                current_row_parts.append(" & ")

                current_row_parts.append(f"{method_name}")
                current_row_parts.append(f" & {data_cells_str} \\\\")
                table_parts.append("".join(current_row_parts))

            if dataset_idx < num_datasets - 1:
                table_parts.append(rf"            \cmidrule(lr){{2-9}}")

        if model_idx < len(model_keys) - 1:
            table_parts.append(r"            \midrule")

    table_parts.append(r"            \bottomrule")
    table_parts.append(r"        \end{tabular}%")
    table_parts.append(r"    }")
    table_parts.append(r"\end{table*}")

    return "\n".join(table_parts)


def hoi_by_split_table_latex(accuracies: dict) -> str:
    def format_row(data_list: list) -> tuple[str, float]:
        avg = sum(data_list) / len(data_list)
        vals_str = " & ".join([f"{v:.1f}" for v in data_list])
        return vals_str, avg

    latex_parts = [
        r'\begin{table}[htbp]',
        r'    \centering',
        r'    \caption{Performance on HOI - detailed breakdown}',
        r'    \label{tab:hoi_lora_comparison_appendix}',
        r'    \resizebox{\textwidth}{!}{',
        r'        \begin{tabular}{@{}l l l c c c c c @{}}',
        r'            \toprule',
        r'''            Model
            & Method
            & Objective
            & \begin{tabular}[c]{@{}c@{}}
                  Seen obj. \\ Seen act.~(\%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  Seen obj. \\ Unseen act.~(\%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  Unseen obj. \\ Seen act.~(\%)
            \end{tabular}
            & \begin{tabular}[c]{@{}c@{}}
                  Unseen obj. \\ Unseen act.~(\%)
            \end{tabular}
            & Avg acc.~(\%) \\''',
        r'            \midrule'
    ]

    model_map = {
        'phi': 'Phi',
        'pixtral': 'Pixtral',
        'gemma3_4b': 'Gemma3 4B'
    }

    row_definitions = [
        ('Baseline gen.', '', 'gen', 0),
        ('Baseline sim.', '', 'sim', 0),
        ('Prompt tuning gen.', '', 'gen', 1),
        ('Prompt tuning sim.', '', 'sim', 1),
        ('Prompt tuning gen.', r'$\mathcal{L}_{\text{sim}}$', 'gen', 2),
        ('Prompt tuning sim.', r'$\mathcal{L}_{\text{sim}}$', 'sim', 2),
        ('LoRA gen.', r'$\mathcal{L}_{\text{NT}}$', 'gen', 3),
        ('LoRA sim.', r'$\mathcal{L}_{\text{NT}}$', 'sim', 3),
        ('LoRA gen.', r'$\mathcal{L}_{\text{combined}}$', 'gen', 4),
        ('LoRA sim.', r'$\mathcal{L}_{\text{combined}}$', 'sim', 4),
    ]

    for model_key in model_map:
        model_name = model_map[model_key]
        gen_data, sim_data = accuracies[model_key]['hoi']
        for i, (method, objective, source, data_idx) in enumerate(row_definitions):
            data_source = gen_data if source == 'gen' else sim_data
            vals_str, avg = format_row(data_source[data_idx][0])
            method_padded = f'{method:<13}'
            objective_padded = f'{objective:<33}'
            if i == 0:
                line = fr'            {model_name}'
            else:
                line = '           '
            line += fr' & {method_padded} & {objective_padded} & {vals_str} & {avg:.1f} \\'
            latex_parts.append(line)
        latex_parts.append(r'            \midrule')
    latex_parts.extend([
        r'            CLIP-RN50',
        r'            & TPT           &                                 & 66.4 & 68.5 & 66.0 & 65.5 & 66.6 \\',
        r'            & SVM-Mimic     &                                 & 69.6 & 70.8 & 78.1 & 71.2 & 72.5 \\',
    ])
    latex_parts.extend([
        r'            \bottomrule',
        r'        \end{tabular}',
        r'    }',
        r'\end{table}'
    ])
    return '\n'.join(latex_parts)


def peft_table_latex(accuracies, methods):
    method_display_names = ["Direct baseline"] + methods

    models_order = [("phi", "Phi"), ("pixtral", "Pixtral"), ("gemma3_4b", "Gemma3 4B")]
    datasets_order = [("openworld", "OpenWorld"), ("hoi", "HOI")]

    latex_output = []
    latex_output.append("\\begin{table*}[htbp]")
    latex_output.append("    \\centering")
    latex_output.append("    \\caption{Summary of PEFT performance on Bongard tasks.}")
    latex_output.append("    \\label{tab:peft_summary}")
    latex_output.append("    \\resizebox{\\textwidth}{!}{%")
    latex_output.append("        \\begin{tabular}{@{}l l l c c c c c@{}}")
    latex_output.append("            \\toprule")

    header_row = (
        "            Model & Dataset & Method & "
        "\\begin{tabular}[c]{@{}c@{}}Generative \\\\ (ID, \\%)\\end{tabular} & "
        "\\begin{tabular}[c]{@{}c@{}}Generative \\\\ (OOD, \\%)\\end{tabular} & "
        "\\begin{tabular}[c]{@{}c@{}}LSC \\\\ (\\%)\\end{tabular} & "
        "\\begin{tabular}[c]{@{}c@{}}Final rep. \\\\ (ID, \\%)\\end{tabular} & "
        "\\begin{tabular}[c]{@{}c@{}}Final rep. \\\\ (OOD, \\%)\\end{tabular} \\\\"
    )
    latex_output.append(header_row)
    latex_output.append("            \\midrule")

    for i_model, (model_key, model_name_display) in enumerate(models_order):
        for i_dataset, (dataset_key, dataset_name_display) in enumerate(datasets_order):
            current_model_dataset_has_accuracy = model_key in accuracies and dataset_key in accuracies[model_key]
            for i_method, method_display_name in enumerate(method_display_names):
                cell1_model = ""
                if i_dataset == 0 and i_method == 0:
                    cell1_model = model_name_display
                cell2_dataset = ""
                if i_method == 0:
                    cell2_dataset = dataset_name_display

                data_cells_str = ["N/A"] * 5
                if current_model_dataset_has_accuracy:
                    try:
                        gen_acc = accuracies[model_key][dataset_key][0][i_method][0]
                        gen_labeled_acc = accuracies[model_key][dataset_key][0][i_method][1]
                        sim_vision_acc = accuracies[model_key][dataset_key][1][i_method][0]
                        sim_final_acc = accuracies[model_key][dataset_key][1][i_method][1]
                        sim_final_labeled_acc = accuracies[model_key][dataset_key][1][i_method][2]

                        data_cells_str = [
                            f"${gen_acc}$",
                            f"${gen_labeled_acc}$",
                            f"${sim_vision_acc}$",
                            f"${sim_final_acc}$",
                            f"${sim_final_labeled_acc}$"
                        ]
                    except (IndexError, TypeError) as e:
                        print(f"Warning: Data format issue for {model_key}/{dataset_key}/Method'{method_display_name}': {e}. Using N/A.")

                all_cells = [cell1_model, cell2_dataset, method_display_name] + data_cells_str
                row_string = " & ".join(all_cells)
                latex_output.append(f"            {row_string} \\\\")

            if i_dataset < len(datasets_order) - 1:
                latex_output.append(f"            \\cmidrule(lr){{2-8}}")

        if i_model < len(models_order) - 1:
            latex_output.append("            \\midrule")

    latex_output.append("            \\bottomrule")
    latex_output.append("        \\end{tabular}%")
    latex_output.append("    }")
    latex_output.append("\\end{table*}")
    return "\n".join(latex_output)


def plot(baseline_scores):
    df = pd.DataFrame(baseline_scores)
    model_display_names = {"phi": "Phi 3.5 vision", "pixtral": "Pixtral 12B", "gemma_4b": "Gemma3 4B", "gemma_27b": "Gemma3 27B", "intern_14b": "InternVL 14B", "qwen_7b": "Qwen2.5-VL 7B", "qwen_72b": "Qwen2.5-VL 72B"}
    dataset_display_names = {
        "openworld": "OpenWorld",
        "hoi": "human-object interaction"
    }
    df['model'] = df['model'].map(model_display_names).fillna(df['model'])
    df['dataset'] = df['dataset'].map(dataset_display_names).fillna(df['dataset'])

    def parse_score(score_str):
        if not isinstance(score_str, str) or not score_str.strip():
            return np.nan
        try:
            return float(score_str.split(' ')[0])
        except (ValueError, IndexError):
            return np.nan

    for col in ['gen_direct', 'gen_cot', 'sim_vision']:
        df[col] = df[col].apply(parse_score)

    df_plot = df.melt(
        id_vars=['model', 'dataset', 'sim_vision'],
        value_vars=['gen_direct', 'gen_cot'],
        var_name='Prompting Method',
        value_name='Generative Accuracy'
    )
    df_plot['Prompting Method'] = df_plot['Prompting Method'].str.replace('gen_', '').str.title()
    df_plot.dropna(subset=['sim_vision', 'Generative Accuracy'], inplace=True)
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    fig.suptitle('Generative performance vs. linear separability', fontsize=16, fontweight='bold')
    all_vals = df_plot['sim_vision'].tolist() + df_plot['Generative Accuracy'].tolist()
    lim_min = np.nanmin(all_vals) - 2
    lim_max = 100

    for i, dataset_name in enumerate(dataset_display_names.values()):
        ax = axes[i]
        data_subset = df_plot[df_plot['dataset'] == dataset_name]

        sns.scatterplot(
            data=data_subset, x='sim_vision', y='Generative Accuracy',
            hue='model', style='Prompting Method', s=150, alpha=0.8, palette='colorblind', ax=ax)

        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.7, zorder=0)
        ax.set_title(dataset_name.title(), fontsize=12)
        ax.set_xlabel('linear classification accuracy (%)', fontsize=12)
        if i == 0:
            ax.set_ylabel('generative accuracy (%)', fontsize=14)

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal', adjustable='box')

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    fig.legend(
        handles, labels,
        fontsize=12,
        loc='lower center',
        bbox_to_anchor=(0.5, 0),
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()


scores, ceiling = agreement_baselines()
print(ceiling)
print(scores)
plot(scores)
print(baseline_table_latex(scores))

accuracies = imbalances_PEFT()
print(accuracies)
print(class_imbalance_table_latex(accuracies))

accuracies, methods = accuracies_PEFT(phi_ablation=True)
print(accuracies)
print(peft_table_latex(accuracies, methods))

accuracies, methods = accuracies_PEFT(highlight=True)
print(accuracies)
print(peft_table_latex(accuracies, methods))

accuracies, methods = accuracies_PEFT(no_vision=True, brief=True, highlight=False)
print(accuracies)
print(peft_table_latex(accuracies, methods))

accuracies, methods = accuracies_PEFT(cross_domain=True, highlight=False, brief=True)
print(accuracies)
print(peft_table_latex(accuracies, methods))

accuracies = splits_PEFT()
print(accuracies)
print(hoi_by_split_table_latex(accuracies))
