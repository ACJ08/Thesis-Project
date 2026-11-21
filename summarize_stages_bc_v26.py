"""Report validation differences between completed arms on the same seed."""
import argparse
import csv
import json
from pathlib import Path

PAIRS = (
    ("architecture", "mean", "native"),
    ("architecture", "cnn", "mean"),
    ("architecture", "lstm", "mean"),
    ("architecture", "cnn_lstm", "mean"),
    ("architecture", "cnn_lstm", "cnn"),
    ("architecture", "cnn_lstm", "lstm"),
    ("architecture", "cnn_lstm", "native"),
    ("supervision", "song_loss", "native"),
    ("context", "longformer_long", "longformer_short"),
    ("context", "longformer_short", "native"),
    ("context", "longformer_long", "native"),
)


def matched_deltas(root):
    for seed_dir in sorted(root.glob("seed_*")):
        results = {
            arm.parent.name: json.loads((arm.parent / "validation_metrics.json").read_text())["metrics"]
            for arm in seed_dir.glob("*/completed.json")
        }
        for experiment, treatment, reference in PAIRS:
            if treatment not in results or reference not in results:
                continue
            for rule, groups in results[treatment].items():
                for group, stats in groups.items():
                    baseline = results[reference][rule][group]
                    if baseline["support"] != stats["support"]:
                        raise ValueError(f"Different validation support in {seed_dir}, {treatment}, {reference}")
                    yield {
                        "seed": int(seed_dir.name.removeprefix("seed_")), "experiment": experiment,
                        "arm": treatment, "reference": reference, "rule": rule, "length_group": group,
                        "songs": sum(stats["support"]), "macro_f1_delta": stats["macro_f1"] - baseline["macro_f1"],
                        **{f"{label}_f1_delta": stats["f1"][i] - baseline["f1"][i]
                           for i, label in enumerate(("safe", "mild", "explicit"))},
                        **{f"{label}_recall_delta": stats["recall"][i] - baseline["recall"][i]
                           for i, label in enumerate(("safe", "mild", "explicit"))},
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = list(matched_deltas(args.root))
    if not rows:
        print("No completed matched pairs")
        return
    output = args.root / "paired_deltas.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} within-seed validation comparisons to {output}")


if __name__ == "__main__":
    main()
