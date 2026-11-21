"""Run matched Stage B/C lyric experiments, with epoch-level recovery."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import lyric_classifier_v26_all_models as v26

ARMS = {
    "native": ("native", "window", 512),
    "mean": ("mean", "window", 512),
    "cnn": ("cnn", "window", 512),
    "lstm": ("lstm", "window", 512),
    "cnn_lstm": ("cnn_lstm", "window", 512),
    "song_loss": ("native", "song", 512),
    "longformer_short": ("longformer", "window", 512),
    "longformer_long": ("longformer", "window", 4096),
}
ROOT = Path(__file__).resolve().parent
BACKBONE = ROOT / "experiments/backbones/longformer-base-4096"
RUN = ROOT / "experiments/stages_bc_v26"
SEEDS = (67, 101, 202)
SONGS_PER_STEP = 8
MODEL_KWARGS = {"num_labels": 3, "id2label": v26.ID2LABEL, "label2id": v26.LABEL2ID}


class CustomHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("roberta-base", local_files_only=True)
        self.projection = nn.Linear(self.encoder.config.hidden_size, 256)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 3))
        self.cnn = None
        self.lstm = None
        if head in ("cnn", "cnn_lstm"):
            self.cnn = nn.ModuleList([nn.Conv1d(256, 256, k, padding=k // 2) for k in (3, 5)])
        if head in ("lstm", "cnn_lstm"):
            self.lstm = nn.LSTM(256, 256, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids, attention):
        mask = attention.unsqueeze(-1)
        hidden = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
        hidden = self.projection(hidden) * mask
        if self.cnn is not None:
            for conv in self.cnn:
                hidden = self.dropout(conv(hidden.transpose(1, 2)).transpose(1, 2).relu()) * mask
        if self.lstm is not None:
            lengths = attention.sum(dim=1).cpu()
            with torch.autocast(device_type=ids.device.type, enabled=False):
                packed = pack_padded_sequence(hidden.float(), lengths, batch_first=True, enforce_sorted=False)
                packed, _ = self.lstm(packed)
                hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=ids.shape[1])
        return self.classifier((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))


def make_model(head):
    if head == "native":
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", local_files_only=True, **MODEL_KWARGS
        )
    elif head == "longformer":
        model = AutoModelForSequenceClassification.from_pretrained(
            BACKBONE, local_files_only=True, **MODEL_KWARGS
        )
    else:
        model = CustomHead(head)
    encoder = model.encoder if isinstance(model, CustomHead) else model
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(v26.DEVICE)


def tokenize(frame, tokenizer, context):
    prefix, suffix = v26.special_token_prefix_suffix(tokenizer)
    capacity = context - len(prefix) - len(suffix)
    step = capacity - v26.CHUNK_OVERLAP_TOKENS
    songs = []
    for first in range(0, len(frame), 256):
        texts = frame.text.iloc[first:first + 256].tolist()
        for tokens in tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]:
            windows = []
            for start in range(0, max(1, len(tokens)), step):
                end = min(len(tokens), start + capacity)
                windows.append(torch.tensor(prefix + tokens[start:end] + suffix, dtype=torch.long))
                if end == len(tokens):
                    break
            songs.append((len(tokens) + len(prefix) + len(suffix), windows))
    return songs


def logits_for_songs(model, songs, indices, tokenizer, forward_windows, longformer):
    windows = [window for index in indices for window in songs[index][1]]
    lengths = [len(songs[index][1]) for index in indices]
    batches = []
    for start in range(0, len(windows), forward_windows):
        part = windows[start:start + forward_windows]
        ids = pad_sequence(part, batch_first=True, padding_value=tokenizer.pad_token_id).to(v26.DEVICE)
        sizes = torch.tensor([len(window) for window in part], device=v26.DEVICE)
        attention = (torch.arange(ids.shape[1], device=v26.DEVICE) < sizes[:, None]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if isinstance(model, CustomHead):
                logits = model(ids, attention)
            elif longformer:
                global_attention = torch.zeros_like(attention)
                global_attention[:, 0] = 1
                logits = model(input_ids=ids, attention_mask=attention,
                               global_attention_mask=global_attention).logits
            else:
                logits = model(input_ids=ids, attention_mask=attention).logits
        batches.append(logits.float())
    return torch.cat(batches).split(lengths)


def mean_log_probability(logits):
    return torch.logsumexp(F.log_softmax(logits, dim=1), dim=0) - math.log(len(logits))


def train_epoch(model, songs, labels, weights, tokenizer, arm, optimizer, scheduler, seed, epoch):
    head, objective, _ = ARMS[arm]
    longformer = head == "longformer"
    forward_windows = 1 if longformer else 4
    v26.set_seed(seed + 10000 * epoch)
    order = np.random.default_rng(seed + 10000 * epoch).permutation(len(songs))
    model.train()
    total_loss = 0.0
    steps = math.ceil(len(songs) / SONGS_PER_STEP)
    for step, first in enumerate(range(0, len(songs), SONGS_PER_STEP), 1):
        ids = order[first:first + SONGS_PER_STEP]
        target = labels[ids]
        grouped = logits_for_songs(model, songs, ids, tokenizer, forward_windows, longformer)
        if objective == "song":
            per_song = F.nll_loss(torch.stack([mean_log_probability(x) for x in grouped]), target, reduction="none")
        else:
            song_labels = target.unbind()
            per_song = torch.stack([
                F.cross_entropy(logits, label.expand(len(logits))) for logits, label in zip(grouped, song_labels)
            ])
        weight = weights[target]
        loss = (per_song * weight).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach().item()
        if step == 1 or step % 250 == 0:
            print(f"{arm} seed {seed} epoch {epoch} step {step}/{steps} loss {total_loss / step:.4f}", flush=True)
    return total_loss / steps


@torch.inference_mode()
def evaluate(model, songs, actual, tokenizer, arm, reference_lengths):
    maximum, mean = [], []
    longformer = ARMS[arm][0] == "longformer"
    model.eval()
    for first in range(0, len(songs), SONGS_PER_STEP):
        ids = range(first, min(first + SONGS_PER_STEP, len(songs)))
        groups = logits_for_songs(model, songs, ids, tokenizer, 1 if longformer else 4, longformer)
        maximum.extend(torch.stack([x.argmax(dim=1).max() for x in groups]).cpu().tolist())
        mean.extend(torch.stack([mean_log_probability(x).argmax() for x in groups]).cpu().tolist())
    groups = {"all": reference_lengths > 0, "up_to_512": reference_lengths <= 512,
              "513_to_4096": (reference_lengths > 512) & (reference_lengths <= 4096),
              "over_4096": reference_lengths > 4096}
    summary = {}
    for rule, predictions in (("max_severity", maximum), ("mean_probability", mean)):
        predicted = np.asarray(predictions)
        summary[rule] = {}
        for group, mask in groups.items():
            if not mask.any():
                continue
            p, r, f, support = precision_recall_fscore_support(
                actual[mask], predicted[mask], labels=range(3), zero_division=0
            )
            summary[rule][group] = {
                "macro_f1": float(f.mean()), "precision": p.tolist(), "recall": r.tolist(),
                "f1": f.tolist(), "support": support.tolist(),
                "confusion_matrix": confusion_matrix(actual[mask], predicted[mask], labels=range(3)).tolist(),
            }
    return summary, pd.DataFrame({"validation_row": range(len(actual)), "true_id": actual,
                                  "roberta_tokens": reference_lengths,
                                  "max_severity": maximum, "mean_probability": mean})


def load_split(name, filename, expected_hash):
    path = ROOT / "data" / filename
    if v26.file_sha256(path) != expected_hash:
        raise ValueError(f"{name} data differs from the matched November run")
    frame = v26.normalize_dataset_columns(pd.read_csv(path, low_memory=False), name.title() + " dataset")
    frame["text"] = frame.text.apply(v26.clean_lyrics)
    frame["label"] = frame.explicitness.astype(str).str.strip().str.lower().map(v26.LABEL_NORMALIZATION).map(v26.LABEL2ID)
    frame = frame.loc[frame.text.ne("") & frame.label.notna()].reset_index(drop=True)
    frame["label"] = frame.label.astype(int)
    return frame


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def select_smoke(frame, tokenizer, per_class):
    short = frame.groupby("label", sort=True).head(per_class)
    lengths = frame.text.str.len()
    longest = frame.loc[lengths.nlargest(1).index]
    candidates = frame.loc[lengths.between(2200, 3400)].head(40)
    token_counts = [len(v26.full_token_ids(text, tokenizer)) for text in candidates.text]
    valid = [idx for idx, count in zip(candidates.index, token_counts) if count > 510]
    return pd.concat([short, longest, frame.loc[valid[:1]]]).loc[lambda x: ~x.index.duplicated()].reset_index(drop=True)


def fit(arm, seed, train, validation, tokenized, reference_lengths, tokenizer, epochs, root, hashes):
    head, objective, context = ARMS[arm]
    output = root / f"seed_{seed}" / arm
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "arm": arm, "head": head, "objective": objective, "context": context,
        "input_sha256": hashes, "training_rows": len(train), "validation_rows": len(validation),
        "seed": seed, "epochs": epochs, "songs_per_step": SONGS_PER_STEP, "overlap": v26.CHUNK_OVERLAP_TOKENS,
        "learning_rate": v26.BASELINE_LEARNING_RATE, "warmup_ratio": v26.WARMUP_RATIO,
        "selection": "maximum-severity validation macro F1", "torch": torch.__version__,
        "transformers": v26.installed_version("transformers"),
        "runner_sha256": v26.file_sha256(Path(__file__)),
        "backbone": "allenai/longformer-base-4096" if head == "longformer" else "roberta-base",
    }
    config_path = output / "manifest.json"
    if config_path.exists() and json.loads(config_path.read_text()) != manifest:
        raise ValueError(f"Configuration changed for {output}")
    write_json(config_path, manifest)
    completed = output / "completed.json"
    if completed.exists():
        print(f"Already complete: {arm} seed {seed}", flush=True)
        return
    training_songs, validation_songs = tokenized
    v26.set_seed(seed)
    model = make_model(head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=v26.BASELINE_LEARNING_RATE)
    total = math.ceil(len(train) / SONGS_PER_STEP) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(v26.WARMUP_RATIO * total), total)
    targets = torch.tensor(train.label.tolist(), device=v26.DEVICE)
    counts = targets.bincount(minlength=3)
    if (counts == 0).any():
        raise ValueError("Training split lacks a severity class")
    weights = len(train) / (3 * counts.float())
    history, best = [], -1.0
    latest = output / "latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best = saved["history"], saved["best"]
        del saved
    for epoch in range(len(history) + 1, epochs + 1):
        loss = train_epoch(model, training_songs, targets, weights, tokenizer, arm,
                           optimizer, scheduler, seed, epoch)
        summary, predictions = evaluate(model, validation_songs, validation.label.to_numpy(),
                                        tokenizer, arm, reference_lengths)
        score = summary["max_severity"]["all"]["macro_f1"]
        print(f"{arm} seed {seed} epoch {epoch} validation macro F1 {score:.4f}", flush=True)
        history.append({"epoch": epoch, "train_loss": loss, "validation_macro_f1": score})
        if score > best:
            best = score
            save_state(output / "best.pt", model.state_dict())
            write_json(output / "validation_metrics.json", {"epoch": epoch, "metrics": summary})
            predictions.to_csv(output / "validation_predictions.csv", index=False)
        save_state(latest, {"history": history, "best": best, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
    write_json(completed, {"history": history, "best_macro_f1": best})
    latest.unlink(missing_ok=True)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide positive epochs and unique seeds")
    if v26.DEVICE.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This campaign requires CUDA with BF16 support")
    frozen = ROOT / "experiments/roberta_v26_20261117T162744Z/manifest.json"
    hashes = json.loads(frozen.read_text())["input_sha256"]
    training = load_split("training", "Training_Cleaned.csv", hashes["training"])
    validation = load_split("validation", "Validation_Cleaned.csv", hashes["validation"])
    roberta = AutoTokenizer.from_pretrained("roberta-base", local_files_only=True)
    if args.smoke:
        training, validation = select_smoke(training, roberta, 2), select_smoke(validation, roberta, 2)
    epochs = 1 if args.smoke else args.epochs
    root = RUN / ("smoke" if args.smoke else "full")
    root.mkdir(parents=True, exist_ok=True)
    roberta_data = tokenize(training, roberta, 512), tokenize(validation, roberta, 512)
    reference_lengths = np.asarray([item[0] for item in roberta_data[1]])
    cache = {("roberta", 512): roberta_data}
    tokenizers = {"roberta": roberta}
    seeds = args.seeds[:1] if args.smoke else args.seeds
    # Stage B and supervision complete before the two context controls.
    failures = {}
    for arm in args.arms:
        head, _, context = ARMS[arm]
        family = "longformer" if head == "longformer" else "roberta"
        try:
            if family not in tokenizers:
                tokenizers[family] = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
            tokenizer = tokenizers[family]
            if (family, context) not in cache:
                cache[(family, context)] = tokenize(training, tokenizer, context), tokenize(validation, tokenizer, context)
            for seed in seeds:
                fit(arm, seed, training, validation, cache[(family, context)], reference_lengths,
                    tokenizer, epochs, root, hashes)
        except Exception as error:
            failures[arm] = repr(error)
            print(f"ERROR {arm}: {error!r}", flush=True)
            if args.smoke:
                raise
    records = []
    for arm in args.arms:
        for seed in seeds:
            directory = root / f"seed_{seed}" / arm
            if (directory / "completed.json").exists():
                best = json.loads((directory / "validation_metrics.json").read_text())
                records.append({"arm": arm, "seed": seed, "epoch": best["epoch"],
                                **{f"{rule}_{group}_macro_f1": stats["macro_f1"]
                                   for rule, groups in best["metrics"].items() for group, stats in groups.items()}})
    pd.DataFrame(records).to_csv(root / "comparison.csv", index=False)
    write_json(root / "campaign_status.json", {"planned": len(args.arms) * len(seeds),
                                               "completed": len(records), "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
