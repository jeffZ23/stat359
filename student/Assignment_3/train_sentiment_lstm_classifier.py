import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
import gensim.downloader as api
from gensim.models import KeyedVectors

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?", text)


def load_fasttext(name: str = "fasttext-wiki-news-subwords-300") -> KeyedVectors:
    return api.load(name)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


def load_phrasebank(dataset_name: str, dataset_config: str) -> Tuple[List[str], np.ndarray]:
    ds = load_dataset(dataset_name, dataset_config)
    train = ds["train"]
    texts = list(train["sentence"])
    labels = np.array(train["label"], dtype=np.int64)
    return texts, labels


def stratified_splits(
    texts: List[str],
    labels: np.ndarray,
    seed: int,
    test_size: float,
    val_size_from_trainval: float,
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    y_trainval = labels[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_size_from_trainval, random_state=seed, stratify=y_trainval
    )

    def pick(ii: np.ndarray):
        return [texts[i] for i in ii], labels[ii]

    x_tr, y_tr = pick(idx_train)
    x_va, y_va = pick(idx_val)
    x_te, y_te = pick(idx_test)
    return {"train": (x_tr, y_tr), "val": (x_va, y_va), "test": (x_te, y_te)}


def text_to_padded_vectors(text: str, ft: KeyedVectors, seq_len: int, dim: int) -> np.ndarray:
    toks = tokenize(text)
    out = np.zeros((seq_len, dim), dtype=np.float32)
    j = 0
    for t in toks:
        if t in ft:
            out[j] = ft[t]
            j += 1
            if j >= seq_len:
                break
    return out


def precompute_sequences(texts: List[str], ft: KeyedVectors, seq_len: int, dim: int) -> np.ndarray:
    out = np.zeros((len(texts), seq_len, dim), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = text_to_padded_vectors(t, ft, seq_len=seq_len, dim=dim)
    return out


class SequenceVecDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(sequences).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        bidirectional: bool,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        h = self.dropout(h)
        return self.fc(h)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_y.append(y.numpy())
    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"acc": float(acc), "macro_f1": float(f1)}


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total += float(loss.item()) * bs
        n += bs
    return total / max(1, n)


@torch.no_grad()
def predict_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_y.append(y.numpy())
    return np.concatenate(all_y, axis=0), np.concatenate(all_preds, axis=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: Any,
    amp: bool,
    grad_clip_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_amp = amp and (device.type in {"cuda", "mps"})
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


def plot_curves(out_dir: str, history: Dict[str, Any]) -> None:
    epochs = history["epoch"]

    def do_plot(key: str, fname: str, ylabel: str):
        plt.figure()
        plt.plot(epochs, history[f"train_{key}"], label="train")
        plt.plot(epochs, history[f"val_{key}"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    do_plot("loss", "curves_loss.png", "Loss")
    do_plot("acc", "curves_accuracy.png", "Accuracy")
    do_plot("macro_f1", "curves_macro_f1.png", "Macro F1")


def plot_confusion_matrix(out_path: str, cm: np.ndarray, labels: List[str]) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


@dataclass
class TrainConfig:
    dataset_name: str = "financial_phrasebank"
    dataset_config: str = "sentences_50agree"
    seed: int = 42
    test_size: float = 0.15
    val_size_from_trainval: float = 0.15

    seq_len: int = 32
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True

    hidden_size: int = 256
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3

    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 40
    min_epochs: int = 30
    patience: int = 8

    grad_clip_norm: float = 1.0
    amp: bool = True

    out_dir: str = "artifacts/lstm_fasttext"
    best_path: str = "artifacts/lstm_fasttext/best_lstm_fasttext.pt"


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = resolve_device()

    out_dir = ensure_dir(cfg.out_dir)

    print("\n========== Loading Dataset ==========")
    texts, labels = load_phrasebank(cfg.dataset_name, cfg.dataset_config)
    print(f"Loaded: n={len(texts)}")

    splits = stratified_splits(
        texts=texts,
        labels=labels,
        seed=cfg.seed,
        test_size=cfg.test_size,
        val_size_from_trainval=cfg.val_size_from_trainval,
    )
    x_tr, y_tr = splits["train"]
    x_va, y_va = splits["val"]
    x_te, y_te = splits["test"]

    num_classes = int(np.max(labels)) + 1
    print("\n========== Split Sizes ==========")
    print(f"Train: {len(x_tr)}  Val: {len(x_va)}  Test: {len(x_te)}")
    print("Train class counts:", np.bincount(y_tr, minlength=num_classes).tolist())
    print("Val class counts:", np.bincount(y_va, minlength=num_classes).tolist())
    print("Test class counts:", np.bincount(y_te, minlength=num_classes).tolist())

    print("\n========== Loading FastText ==========")
    ft = load_fasttext("fasttext-wiki-news-subwords-300")
    dim = int(ft.vector_size)

    print("\n========== Precomputing Sequences ==========")
    tr_seq = precompute_sequences(x_tr, ft, seq_len=cfg.seq_len, dim=dim)
    va_seq = precompute_sequences(x_va, ft, seq_len=cfg.seq_len, dim=dim)
    te_seq = precompute_sequences(x_te, ft, seq_len=cfg.seq_len, dim=dim)

    train_ds = SequenceVecDataset(tr_seq, y_tr)
    val_ds = SequenceVecDataset(va_seq, y_va)
    test_ds = SequenceVecDataset(te_seq, y_te)

    pin_memory = bool(cfg.pin_memory) if device.type != "mps" else False

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=False)

    class_w = compute_class_weights(y_tr, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    model = LSTMClassifier(
        input_dim=dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        num_classes=num_classes,
        bidirectional=cfg.bidirectional,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_macro_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    bad = 0

    print("\n========== Training ==========")
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler if (cfg.amp and device.type == "cuda") else None,
            amp=cfg.amp,
            grad_clip_norm=cfg.grad_clip_norm if cfg.grad_clip_norm and cfg.grad_clip_norm > 0 else None,
        )

        train_metrics = evaluate_metrics(model, train_loader, device)
        val_metrics = evaluate_metrics(model, val_loader, device)

        val_loss = evaluate_loss(model, val_loader, device, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_metrics["acc"]))
        history["train_macro_f1"].append(float(train_metrics["macro_f1"]))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_metrics["acc"]))
        history["val_macro_f1"].append(float(val_metrics["macro_f1"]))

        print(json.dumps({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_metrics["acc"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_loss),
            "val_acc": float(val_metrics["acc"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "num_classes": num_classes,
                    "ft_name": "fasttext-wiki-news-subwords-300",
                    "best_epoch": best_epoch,
                    "val_macro_f1": best_val_f1,
                },
                cfg.best_path,
            )
        else:
            if epoch >= cfg.min_epochs:
                bad += 1

        save_json(os.path.join(out_dir, "history.json"), history)

        if epoch >= cfg.min_epochs and bad >= cfg.patience:
            break

    plot_curves(out_dir, history)

    print("\n========== Best Model Evaluation ==========")
    ckpt = torch.load(cfg.best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = evaluate_metrics(model, test_loader, device)
    test_loss = evaluate_loss(model, test_loader, device, criterion)

    y_true, y_pred = predict_labels(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    save_json(os.path.join(out_dir, "confusion_matrix.json"), {"labels": ["negative", "neutral", "positive"], "matrix": cm.tolist()})
    plot_confusion_matrix(os.path.join(out_dir, "confusion_matrix.png"), cm, ["negative", "neutral", "positive"])

    print(json.dumps({
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_loss": float(test_loss),
        "test_acc": float(test_metrics["acc"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }))


if __name__ == "__main__":
    main()

# {"best_epoch": 38, "best_val_macro_f1": 0.7097494629140199, "test_loss": 2.303729201937312, "test_acc": 0.7579092159559835, "test_macro_f1": 0.716082403345391}
