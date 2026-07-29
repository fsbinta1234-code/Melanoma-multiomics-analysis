"""
Neural network — BRAFi resistance phosphosite classifier.

Task: predict whether a phosphosite is differentially phosphorylated between
BRAF V600E and NRAS melanoma cell lines (binary classification).

Data (100% real):
  • PXD022992 directDIA phosphoproteome — 6 melanoma cell lines:
      A375, SH4, SK-MEL-28, RPMI-7951  (BRAF V600E)
      G361, SK-MEL-31                  (NRAS mutant)
  • Labels from differential analysis (t-test, p<0.05 AND |log2FC|>1):
      1 = differentially phosphorylated  (2 691 sites)
      0 = not differentially phosphorylated  (2 691 balanced, sampled)

Architecture (MLP):
  6 → BN → 128 → BN → ReLU → Dropout(0.3) →
       64 → BN → ReLU → Dropout(0.3) →
       32 → ReLU →
        1 (BCEWithLogitsLoss)

Training:
  • Stratified 70/30 train/test split
  • AdamW + CosineAnnealingLR, 120 epochs
  • Tracked per epoch: train loss, test loss, train AUC, test AUC
  • Figures: loss curves + AUC curves (same canvas), ROC curve, confusion matrix

Saved outputs (results/):
  • figures/nn_training_curves.png       — loss + AUC vs epoch
  • figures/nn_roc_confusion.png         — ROC + confusion matrix
  • outputs/nn_training_history.csv      — full epoch log
  • outputs/nn_predictions_test.csv      — per-site probability on test set
"""
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, average_precision_score,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import pipeline_config as cfg

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── hyperparameters ───────────────────────────────────────────────────────────
EPOCHS      = 120
BATCH_SIZE  = 128
LR          = 3e-4
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.30

# ── data files ────────────────────────────────────────────────────────────────
MATRIX_FILE = "results/outputs/PXD022992_phosphosite_matrix.csv"
DIFF_FILE   = "results/outputs/PXD022992_differential_phospho_BRAFvsNRAS.csv"

SAMPLE_COLS = ["SK", "HTB69", "G361", "SH4", "A375", "7951"]   # cell lines in matrix


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data preparation
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset():
    matrix = pd.read_csv(MATRIX_FILE, index_col=0)
    diff   = pd.read_csv(DIFF_FILE)
    if "site" in diff.columns:
        diff = diff.set_index("site")

    # intensity columns
    feat_cols = [c for c in SAMPLE_COLS if c in matrix.columns]
    assert feat_cols, "No sample columns found in matrix"

    # Labels: significant differential sites (use site IDs, not numeric indices)
    diff = diff[diff.index.isin(matrix.index)]
    sig    = diff[(diff["pval"] < 0.05) & (diff["log2FC_BRAFvsNRAS"].abs() > 1)].index
    nonsig = diff.index.difference(sig)

    # Balance: sample same number of negatives as positives
    rng = np.random.default_rng(SEED)
    neg_idx = rng.choice(nonsig, size=len(sig), replace=False)
    selected = np.concatenate([sig.values, neg_idx])

    X = matrix.loc[selected, feat_cols].values.astype(np.float32)
    y = np.array([1.0] * len(sig) + [0.0] * len(neg_idx), dtype=np.float32)

    # Drop rows with any NaN
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]

    # Stratified 70/30 split
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.30,
                                      stratify=y.astype(int),
                                      random_state=SEED)

    # Feature normalization fit on TRAIN only
    scaler = StandardScaler()
    X[idx_tr] = scaler.fit_transform(X[idx_tr])
    X[idx_te] = scaler.transform(X[idx_te])

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]

    print(f"[data] total={len(y)}  train={len(y_tr)}  test={len(y_te)}  "
          f"features={X.shape[1]}  positives={int(y.sum())}/{len(y)}")
    # Return the TEST-set site IDs aligned to X_te order (idx_te is the shuffled,
    # stratified test selection — slicing the array by position would misalign).
    site_ids = selected[valid]
    return (X_tr, y_tr), (X_te, y_te), feat_cols, site_ids[idx_te]


def make_loaders(X_tr, y_tr, X_te, y_te):
    def _tensor_ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
    train_dl = DataLoader(_tensor_ds(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_dl  = DataLoader(_tensor_ds(X_te, y_te),
                          batch_size=BATCH_SIZE * 2, shuffle=False)
    return train_dl, test_dl


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model
# ─────────────────────────────────────────────────────────────────────────────
class PhosphositeNet(nn.Module):
    """
    3-hidden-layer MLP with Batch Normalization and Dropout.
    Designed for small tabular feature spaces (6 cell-line intensities).
    """
    def __init__(self, n_features: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training loop
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(model, loader, criterion, device):
    """Return (loss, auc, probs, labels)."""
    model.eval()
    losses, probs_all, labels_all = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            losses.append(criterion(logits, yb).item() * len(xb))
            probs_all.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            labels_all.extend(yb.cpu().numpy().ravel())
    loss = sum(losses) / len(labels_all)
    auc  = roc_auc_score(labels_all, probs_all)
    return loss, auc, np.array(probs_all), np.array(labels_all)


def train(model, train_dl, test_dl, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {k: [] for k in ["epoch",
                                "train_loss", "test_loss",
                                "train_auc",  "test_auc",  "lr"]}

    print(f"\n[train] {sum(p.numel() for p in model.parameters()):,} params | "
          f"device={device} | epochs={EPOCHS}")
    print(f"{'Epoch':>6} {'Train loss':>12} {'Test loss':>11} "
          f"{'Train AUC':>11} {'Test AUC':>10}")
    print("─" * 58)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item() * len(xb))
        train_loss = sum(tr_losses) / len(train_dl.dataset)
        train_auc  = _evaluate(model, train_dl, criterion, device)[1]
        test_loss, test_auc, probs_te, labels_te = _evaluate(
            model, test_dl, criterion, device)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_auc"].append(train_auc)
        history["test_auc"].append(test_auc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}   {train_loss:>10.4f}   {test_loss:>10.4f}"
                  f"   {train_auc:>9.4f}   {test_auc:>9.4f}")

    print("─" * 58)
    return pd.DataFrame(history), probs_te, labels_te


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curves(history):
    """Two-panel figure: loss curves (left) and AUC curves (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"],
            lw=2, color=cfg.COLOR_RESIST,   label="Train loss")
    ax.plot(history["epoch"], history["test_loss"],
            lw=2, color=cfg.COLOR_CONTROL,  label="Test loss",
            linestyle="--")
    ax.fill_between(history["epoch"],
                    history["train_loss"], history["test_loss"],
                    alpha=0.12, color=cfg.COLOR_ACCENT)
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Training and test loss — phosphosite classifier")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)

    # annotate final gap
    final_tr = history["train_loss"].iloc[-1]
    final_te = history["test_loss"].iloc[-1]
    ax.annotate(f"Final train: {final_tr:.4f}",
                xy=(history["epoch"].iloc[-1], final_tr),
                xytext=(-80, 10), textcoords="offset points",
                fontsize=8, color=cfg.COLOR_RESIST,
                arrowprops=dict(arrowstyle="->", color=cfg.COLOR_RESIST, lw=0.8))
    ax.annotate(f"Final test: {final_te:.4f}",
                xy=(history["epoch"].iloc[-1], final_te),
                xytext=(-80, -20), textcoords="offset points",
                fontsize=8, color=cfg.COLOR_CONTROL,
                arrowprops=dict(arrowstyle="->", color=cfg.COLOR_CONTROL, lw=0.8))

    # ── AUC ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(history["epoch"], history["train_auc"],
            lw=2, color=cfg.COLOR_RESIST,   label="Train AUC")
    ax.plot(history["epoch"], history["test_auc"],
            lw=2, color=cfg.COLOR_CONTROL,  label="Test AUC",
            linestyle="--")
    ax.axhline(0.5, color="lightgrey", lw=0.8, ls=":")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Epoch"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Training and test AUC — phosphosite classifier")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)

    best_ep  = history["test_auc"].idxmax()
    best_auc = history["test_auc"].max()
    ax.axvline(history["epoch"].iloc[best_ep], color="dimgrey",
               lw=0.8, ls="--", alpha=0.6)
    ax.annotate(f"Best test AUC\n{best_auc:.4f} @ ep{history['epoch'].iloc[best_ep]}",
                xy=(history["epoch"].iloc[best_ep], best_auc),
                xytext=(10, -25), textcoords="offset points", fontsize=8)

    fig.suptitle(
        "Neural network — predict differentially phosphorylated sites\n"
        "(PXD022992 DIA, BRAF V600E vs NRAS, 6 melanoma cell lines)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    cfg.save_figure(fig, "nn_training_curves")


def plot_roc_confusion(probs_te, labels_te):
    """ROC curve + confusion matrix side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── ROC ───────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels_te, probs_te)
    auc_val      = roc_auc_score(labels_te, probs_te)
    ap_val       = average_precision_score(labels_te, probs_te)

    axes[0].plot(fpr, tpr, lw=2.5, color=cfg.COLOR_RESIST,
                 label=f"ROC-AUC = {auc_val:.4f}\nAP = {ap_val:.4f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="lightgrey", lw=1)
    axes[0].fill_between(fpr, 0, tpr, alpha=0.12, color=cfg.COLOR_RESIST)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve — test set")
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.01)
    axes[0].grid(alpha=0.3)

    # ── Confusion matrix ──────────────────────────────────────────────────
    y_pred = (probs_te >= 0.5).astype(int)
    cm     = confusion_matrix(labels_te.astype(int), y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-diff", "Differential"],
                yticklabels=["Non-diff", "Differential"],
                ax=axes[1], cbar=False,
                linewidths=0.5, linecolor="#ddd")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion matrix — test set (threshold 0.5)")

    report = classification_report(labels_te.astype(int), y_pred,
                                   target_names=["Non-diff", "Differential"])
    print("\n" + report)

    fig.suptitle(
        f"Test set evaluation — ROC-AUC {auc_val:.4f}  |  Average Precision {ap_val:.4f}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    cfg.save_figure(fig, "nn_roc_confusion")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────
def run():
    cfg.apply_style()
    print("=" * 60)
    print("Neural network — phosphosite differential classifier")
    print("=" * 60)

    # Check inputs
    for f in [MATRIX_FILE, DIFF_FILE]:
        if not os.path.exists(f):
            print(f"[ERROR] {f} not found.\n"
                  "        Run pxd022992_dia_phosphoproteome_analysis.py first.")
            return

    # Data
    (X_tr, y_tr), (X_te, y_te), feat_cols, test_site_ids = build_dataset()
    train_dl, test_dl = make_loaders(X_tr, y_tr, X_te, y_te)

    # Model
    device = torch.device("cpu")  # CPU is fine for this size
    model  = PhosphositeNet(n_features=len(feat_cols)).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[model] architecture:\n{model}")

    # Train
    history, probs_te, labels_te = train(model, train_dl, test_dl, device)

    # Plots
    print("\n[plots] Training curves...")
    plot_training_curves(history)
    print("[plots] ROC + confusion matrix...")
    plot_roc_confusion(probs_te, labels_te)

    # Save outputs
    cfg.save_table(history, "nn_training_history.csv")

    # Save per-site predictions on the test set (site IDs aligned to X_te order)
    pred_df = pd.DataFrame({
        "site": test_site_ids[:len(probs_te)],
        "prob_differential": probs_te,
        "predicted": (probs_te >= 0.5).astype(int),
        "true_label": labels_te.astype(int),
    })
    cfg.save_table(pred_df, "nn_predictions_test.csv")

    final_test_auc = history["test_auc"].iloc[-1]
    best_test_auc  = history["test_auc"].max()
    print(f"\n[summary] Final test AUC : {final_test_auc:.4f}")
    print(f"[summary] Best  test AUC : {best_test_auc:.4f}")
    print("\nDone.")


def main():
    run()


if __name__ == "__main__":
    main()
