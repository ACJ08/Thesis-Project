"""Paired inference and figures for annotator_bias_experiment_v26.py."""
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from learning_curve_v26 import read_json

LABELS = ("Safe", "Mild", "Explicit")
CONDITIONS = ("A_to_A", "A_to_B", "B_to_A", "B_to_B")
PREFIX = "annotator_experiment_"
RESAMPLES = 2000


def macro_f1(actual, predicted, weights=None):
    cm = np.bincount(3 * actual + predicted, weights=weights, minlength=9).reshape(3, 3)
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.divide(2 * cm.diagonal(), denominator, out=np.zeros(3),
                           where=denominator != 0).mean())


def paired_test(first, second, seed):
    if first.song_uid.tolist() != second.song_uid.tolist() or not np.array_equal(first.actual_id, second.actual_id):
        raise RuntimeError("Paired comparison requires identical ordered test songs and labels.")
    actual = first.actual_id.to_numpy()
    within, cross = first.predicted_id.to_numpy(), second.predicted_id.to_numpy()
    observed = macro_f1(actual, within) - macro_f1(actual, cross)
    rng = np.random.default_rng(seed)
    bootstrap, extreme = [], 0
    for _ in range(RESAMPLES):
        indices = rng.integers(len(actual), size=len(actual))
        bootstrap.append(macro_f1(actual[indices], within[indices]) - macro_f1(actual[indices], cross[indices]))
        swap = rng.random(len(actual)) < .5
        permuted = macro_f1(actual, np.where(swap, cross, within)) - macro_f1(
            actual, np.where(swap, within, cross))
        extreme += abs(permuted) >= abs(observed) - 1e-12
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {"difference": observed, "ci95_low": low, "ci95_high": high,
            "p_raw": (extreme + 1) / (RESAMPLES + 1)}


def holm(values):
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1., running)
    return adjusted


def overall_interval(predictions, seeds, model):
    """Crossed bootstrap sharing song weights across seeds and resampling seed blocks."""
    prepared, uid_by_annotator = {}, {}
    for target in ("A", "B"):
        uids = sorted({uid for seed in seeds for uid in predictions[
            model, seed, f"{target}_to_{target}"].song_uid})
        uid_by_annotator[target] = uids
        lookup = {uid: i for i, uid in enumerate(uids)}
        for seed in seeds:
            within = predictions[model, seed, f"{target}_to_{target}"]
            other = "B" if target == "A" else "A"
            cross = predictions[model, seed, f"{other}_to_{target}"]
            if within.song_uid.tolist() != cross.song_uid.tolist():
                raise ValueError("Overall contrast prediction alignment failed.")
            prepared[seed, target] = (within.actual_id.to_numpy(), within.predicted_id.to_numpy(),
                                      cross.predicted_id.to_numpy(), np.array([lookup[u] for u in within.song_uid]))
    rng = np.random.default_rng(72301)
    values = []
    for _ in range(RESAMPLES):
        weights = {target: rng.multinomial(len(uids), np.full(len(uids), 1 / len(uids)))
                   for target, uids in uid_by_annotator.items()}
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        contrasts = []
        for seed in sampled_seeds:
            for target in ("A", "B"):
                actual, within, cross, indices = prepared[seed, target]
                contrasts.append(macro_f1(actual, within, weights[target][indices]) -
                                 macro_f1(actual, cross, weights[target][indices]))
        values.append(np.mean(contrasts))
    return np.percentile(values, [2.5, 97.5])


def generate_reports(v, root, models, seeds, smoke):
    predictions, results, per_class, confusion, comparisons = {}, [], [], [], []
    for model in models:
        for seed in seeds:
            for training in ("A", "B"):
                run = root / "runs" / f"seed_{seed}" / model / f"train_{training}"
                meta = read_json(run / "checkpoint" / "training_metadata.json")
                native = read_json(run / "checkpoint" / "manifest.json")
                for target in ("A", "B"):
                    condition = f"{training}_to_{target}"
                    directory = run / "evaluation"
                    prediction = pd.read_csv(directory / f"{condition}_predictions.csv")
                    predictions[model, seed, condition] = prediction
                    saved = read_json(directory / f"{condition}_metrics.json")
                    if saved["checkpoint_sha256"] != native["checkpoint_sha256"]:
                        raise RuntimeError("Evaluation and training checkpoint hashes differ.")
                    scores = saved["metrics"]
                    base = {"model": model, "seed": seed, "condition": condition,
                            "training_annotator": training, "test_annotator": target, "test_songs": len(prediction),
                            "training_songs": native["training_rows"], "validation_songs": native["validation_rows"],
                            "best_epoch": native["best_epoch"], "training_duration_seconds": meta["training_duration_seconds"],
                            "checkpoint_sha256": native["checkpoint_sha256"], "smoke_test": smoke}
                    results.append({**base, **{key: value for key, value in scores.items()
                                              if isinstance(value, (int, float))},
                                    "validation_accuracy": meta["validation"]["accuracy"],
                                    "validation_macro_f1": meta["validation"]["macro_f1"],
                                    **{f"training_count_{label}": native["training_class_counts"][label] for label in LABELS}})
                    for label in LABELS:
                        per_class.append({**base, "label": label, **scores["per_class"][label]})
                    for i, actual in enumerate(LABELS):
                        for j, predicted in enumerate(LABELS):
                            confusion.append({**base, "actual": actual, "predicted": predicted,
                                              "count": scores["confusion_matrix"][i][j]})
            for target in ("A", "B"):
                other = "B" if target == "A" else "A"
                stats = paired_test(predictions[model, seed, f"{target}_to_{target}"],
                                    predictions[model, seed, f"{other}_to_{target}"], seed)
                comparisons.append({"model": model, "seed": seed, "contrast": f"delta_{target}",
                                    "bootstrap_resamples": RESAMPLES, "permutation_resamples": RESAMPLES, **stats})
    paired = pd.DataFrame(comparisons)
    paired["p_holm"] = holm(paired.p_raw.to_numpy())
    paired["holm_family_size"] = len(paired)
    paired["holm_family"] = "all_architectures_all_seeds_delta_A_and_delta_B"
    totals = []
    for (model, seed), group in paired.groupby(["model", "seed"]):
        totals.append({"model": model, "seed": seed, "contrast": "delta_overall",
                       "difference": group.difference.mean()})
    stats = pd.concat([paired, pd.DataFrame(totals)], ignore_index=True)
    summary = []
    for (model, contrast), group in stats.groupby(["model", "contrast"]):
        low = high = None
        if contrast == "delta_overall":
            low, high = overall_interval(predictions, seeds, model)
        summary.append({"model": model, "measure": contrast, "mean": group.difference.mean(),
                        "sd": group.difference.std(ddof=1) if len(group) > 1 else None,
                        "seeds": len(group), "ci95_low": low, "ci95_high": high,
                        "interval_method": "crossed song-cluster and seed-block bootstrap" if low is not None else
                        "individual paired intervals reported separately"})
    result_frame = pd.DataFrame(results)
    for (model, condition), group in result_frame.groupby(["model", "condition"]):
        for metric in ("accuracy", "macro_f1"):
            summary.append({"model": model, "measure": condition + "_" + metric,
                            "mean": group[metric].mean(), "sd": group[metric].std(ddof=1) if len(group) > 1 else None,
                            "seeds": len(group)})
    report = Path(mkdtemp(prefix="report_", dir=root))
    tables = {"results": result_frame, "per_class": pd.DataFrame(per_class),
              "confusion": pd.DataFrame(confusion), "paired_statistics": stats,
              "seed_summary": pd.DataFrame(summary)}
    for name, table in tables.items():
        table.to_csv(report / f"{PREFIX}{name}.csv", index=False)
    make_plots(report / "plots", tables, models, seeds)
    description = [
        "# Controlled cross-annotator generalization",
        "DIAGNOSTIC SMOKE RESULTS" if smoke else "Training 4000, validation 1000, test 986 per annotator and seed.",
        "Each trained checkpoint supplies both target-annotator evaluations. Checkpoint selection uses the training annotator's validation Macro-F1.",
        "Annotator identities and human annotation provenance were supplied by the user. Source hashes are in experiment.json. "
        "Sampling manifests report class prevalence, identifiers and allocation. The overlap report documents exclusions.",
        "## Measured performance", "```csv\n" + result_frame[["model", "seed", "condition", "accuracy", "macro_f1"]].to_csv(index=False) + "```",
        "## Within minus cross contrasts", "```csv\n" + stats[["model", "seed", "contrast", "difference", "p_raw", "p_holm"]].to_csv(index=False) + "```",
        "## Across-seed summaries", "```csv\n" + pd.DataFrame(summary).to_csv(index=False) + "```",
        "Positive contrasts are consistent with a within-annotator generalization advantage. Negative contrasts favor cross-annotator training on that target collection.",
        "Individual paired bootstrap intervals and two-sided paired permutation tests use 2000 resamples. "
        "Holm correction spans every selected architecture, seed and target annotator as one prespecified family. "
        "Delta-overall is descriptive, with a crossed bootstrap sharing song weights across seeds and resampling seed blocks. "
        "These intervals condition on fitted checkpoints and cannot fully represent retraining uncertainty. "
        "With three seeds they are exploratory; with one seed they reflect test-song uncertainty only.",
        "Annotators labeled different song collections. Annotation conventions, class prevalence, song difficulty, "
        "genre/domain, dataset composition and optimization can all contribute. The experiment cannot establish "
        "causal annotator bias or rank annotation quality. Independent labels from both annotators on the same songs are required for that question.",
        "Existing B data previously served as thesis holdouts. This experiment repartitions that pool under the requested protocol. "
        "Earlier pipeline choices may have been informed by those songs.",
    ]
    sampling = read_json(root / f"{PREFIX}sampling_manifest.json") if (
        root / f"{PREFIX}sampling_manifest.json").exists() else None
    if sampling:
        prevalence = []
        for seed, annotators in sampling["sampling"].items():
            for annotator, splits in annotators.items():
                for split, specification in splits.items():
                    counts = specification if split == "pool_class_counts" else specification["class_counts"]
                    total = sum(counts.values())
                    for label in LABELS:
                        prevalence.append({"seed": seed, "annotator": annotator, "split": split,
                                           "label": label, "count": counts[label],
                                           "proportion": counts[label] / total if total else None})
        description.extend(["## Class prevalence", "```csv\n" + pd.DataFrame(prevalence).to_csv(index=False) + "```"])
    (report / f"{PREFIX}summary.md").write_text("\n\n".join(description), encoding="utf-8")
    print(f"Reports saved to {report}", flush=True)


def make_plots(directory, tables, models, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 240})
    colors = ("#236192", "#79a7cc", "#e2a15c", "#9b4a1b")
    x = np.arange(len(models))
    results = tables["results"]
    for metric in ("macro_f1", "accuracy"):
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        for i, condition in enumerate(CONDITIONS):
            group = results.loc[results.condition.eq(condition)].groupby("model")[metric]
            ax.bar(x + (i - 1.5) * .19, group.mean().reindex(models), .19,
                   yerr=group.std().reindex(models).fillna(0), capsize=3,
                   color=colors[i], label=condition.replace("_to_", " → "))
        ax.set(xticks=x, xticklabels=models, ylabel=metric.replace("_", " ").title(), ylim=(0, 1),
               title="Four conditions · n=986 test songs/annotator/seed · bars show seed SD")
        ax.legend(ncol=4)
        fig.savefig(directory / f"conditions_{metric}.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    statistics = tables["paired_statistics"]
    for i, contrast in enumerate(("delta_A", "delta_B", "delta_overall")):
        group = statistics.loc[statistics.contrast.eq(contrast)].groupby("model").difference
        ax.errorbar(x + (i - 1) * .12, group.mean().reindex(models),
                    yerr=group.std().reindex(models).fillna(0), fmt="o", capsize=4, label=contrast)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xticks=x, xticklabels=models, ylabel="Within minus cross Macro-F1",
           title="Paired contrasts · n=986 per target/seed · error bars show seed SD")
    ax.legend()
    fig.savefig(directory / "within_cross_differences.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained", sharey=True)
    for ax, label in zip(axes, LABELS):
        for i, condition in enumerate(CONDITIONS):
            group = tables["per_class"].query("label == @label and condition == @condition").groupby("model").f1
            ax.bar(x + (i - 1.5) * .19, group.mean().reindex(models), .19,
                   yerr=group.std().reindex(models).fillna(0), capsize=2, color=colors[i],
                   label=condition.replace("_to_", " → "))
        ax.set(xticks=x, xticklabels=models, title=label, ylim=(0, 1), ylabel="Per-class F1")
        ax.tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-class F1 · n=986 test songs/annotator/seed · error bars show seed SD")
    fig.savefig(directory / "per_class_f1.png")
    plt.close(fig)
    for model in models:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4), layout="constrained")
        for ax, condition in zip(axes, CONDITIONS):
            cm = tables["confusion"].query("model == @model and condition == @condition").groupby(
                ["actual", "predicted"])["count"].mean().unstack().reindex(index=LABELS, columns=LABELS)
            ax.imshow(cm, cmap="Blues")
            for i in range(3):
                for j in range(3):
                    color = "white" if cm.iloc[i, j] > cm.to_numpy().max() * .6 else "black"
                    ax.text(j, i, f"{cm.iloc[i,j]:.1f}", ha="center", va="center", color=color)
            ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
                   xlabel="Predicted", ylabel="Actual", title=condition.replace("_to_", " → "))
        fig.suptitle(f"{model} · mean counts across {len(seeds)} seeds · n=986 per matrix/seed")
        fig.savefig(directory / f"{model}_confusion.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False, layout="constrained")
    for ax, model in zip(axes.flat, models):
        for condition in CONDITIONS:
            group = results.query("model == @model and condition == @condition")
            ax.plot(group.seed.astype(str), group.macro_f1, "o-", label=condition.replace("_to_", " → "))
        ax.set(title=model, xlabel="Seed", ylabel="Macro-F1")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Seed variability · n=986 test songs/annotator/seed")
    fig.savefig(directory / "seed_variability.png")
    plt.close(fig)
    directions = [(a, b) for a in LABELS for b in LABELS if a != b]
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False, layout="constrained")
    for ax, model in zip(axes.flat, models):
        for condition in CONDITIONS:
            group = tables["confusion"].query("model == @model and condition == @condition")
            cm = group.groupby(["actual", "predicted"])["count"].mean().unstack()
            rates = [cm.loc[a, b] / cm.loc[a].sum() for a, b in directions]
            ax.plot(range(6), rates, "o-", label=condition.replace("_to_", " → "))
        ax.set(title=model, ylabel="Error count / actual-class support",
               xticks=range(6), xticklabels=[a + " → " + b for a, b in directions])
        ax.tick_params(axis="x", rotation=60)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Directional errors · mean counts across seeds · n=986 test songs/annotator/seed")
    fig.savefig(directory / "directional_errors.png")
    plt.close(fig)
