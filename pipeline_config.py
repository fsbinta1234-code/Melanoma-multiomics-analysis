"""
Central configuration for the melanoma multi-omics pipeline.

Gathers in one place:
  * Output paths (results/figures, results/outputs).
  * The EXPERIMENTAL DESIGN (sample -> condition mapping) of all three layers.
  * Helpers to select / group sample columns.
  * A consistent visual style for publication-quality figures + a save helper.
  * Missing-value imputation (Perseus-style) for MS data.

────────────────────────────────────────────────────────────────────────────
EXPERIMENTAL DESIGN (study: "Androgen Receptor is a Determinant of Melanoma
BRAFi/MEKi Resistance", GEO GSE199405 + PRIDE PXD026557)

Androgen receptor (AR) overexpression renders melanoma cells RESISTANT to
BRAFi/MEKi. Therefore, across every layer:
      RESISTANT  ==  ARoe  (AR overexpression)
      CONTROL    ==  LacZ  (control vector)

• Transcriptomics (GEO): 12 samples, mapping confirmed by the !Sample_title
  metadata of the series_matrix file (see GEO_SAMPLE below).

• Phospho/Proteomics (MaxQuant): columns named 1A..12D = 12 samples × 4
  replicates (A–D). The sample number (1–12) is NOT labelled in the file; here
  we assume it follows the SAME condition order as the GEO data (documented
  assumption — verify against the PXD026557 metadata for absolute precision).
────────────────────────────────────────────────────────────────────────────
"""
import os
import re

import matplotlib

# Non-interactive backend: never opens a window nor blocks execution.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
OUT_DIR = os.path.join(RESULTS_DIR, "outputs")


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIG_DIR, OUT_DIR):
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Visual style
# ──────────────────────────────────────────────────────────────────────────
COLOR_CONTROL = "#4C72B0"   # blue — LacZ (control / sensitive)
COLOR_RESIST = "#C44E52"    # red  — ARoe (resistant)
COLOR_NS = "#BDBDBD"        # grey — not significant
COLOR_UP = "#C44E52"        # up-regulated in resistant
COLOR_DOWN = "#4C72B0"      # down-regulated in resistant
COLOR_ACCENT = "#55A868"    # green — highlights
HEATMAP_CMAP = "RdBu_r"

# Per-cell-line colours (for annotating plots)
CELL_LINE_COLORS = {"A375": "#E69F00", "M14": "#009E73", "WM9": "#CC79A7"}


def apply_style() -> None:
    """Apply a consistent, readable visual theme to all figures."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "font.family": "DejaVu Sans",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig, name: str) -> str:
    """Save the figure to results/figures/<name>.png and close it.

    Returns the path of the saved file.
    """
    ensure_dirs()
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [figure] {os.path.relpath(path, BASE_DIR)}")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> str:
    """Save a DataFrame to results/outputs/<name> and return the path."""
    ensure_dirs()
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=index)
    print(f"  [csv]    {os.path.relpath(path, BASE_DIR)}  ({df.shape[0]}×{df.shape[1]})")
    return path


# ──────────────────────────────────────────────────────────────────────────
# Experimental design — MaxQuant (phospho + protein)
# ──────────────────────────────────────────────────────────────────────────
# sample number (1–12) -> (cell line, vector, treatment)
MAXQUANT_SAMPLE = {
    1: ("A375", "ARoe", "Dabrafenib"), 2: ("A375", "ARoe", "DMSO"),
    3: ("A375", "LacZ", "Dabrafenib"), 4: ("A375", "LacZ", "DMSO"),
    5: ("M14", "ARoe", "Dabrafenib"), 6: ("M14", "ARoe", "DMSO"),
    7: ("M14", "LacZ", "Dabrafenib"), 8: ("M14", "LacZ", "DMSO"),
    9: ("WM9", "ARoe", "Dabrafenib"), 10: ("WM9", "ARoe", "DMSO"),
    11: ("WM9", "LacZ", "Dabrafenib"), 12: ("WM9", "LacZ", "DMSO"),
}

# Clean sample token: number (1–12) followed by replicate (A–D), e.g. "5A".
_TOKEN_RE = re.compile(r"^(\d{1,2})([A-D])$")


def maxquant_clean_column_map(columns, prefix: str) -> dict:
    """Map VALID sample-intensity columns (1A..12D) -> sample token.

    Ignores aggregate columns ('Intensity', 'Intensity___1', etc.) and spurious
    variants ('8A_2', '8A_enrichment2'), which do not match the <num><A-D> pattern.

    Parameters
    ----------
    columns : iterable of raw DataFrame column names.
    prefix  : 'Intensity ' (phospho) or 'LFQ intensity ' (protein).
    """
    mapping = {}
    for col in columns:
        if col.startswith(prefix):
            token = col[len(prefix):].strip()
            if _TOKEN_RE.match(token):
                mapping[col] = token
    return mapping


def _token_number(token: str) -> int:
    return int(_TOKEN_RE.match(token).group(1))


def maxquant_groups(matrix: pd.DataFrame):
    """Return (control_columns, resistant_columns) of a matrix whose columns
    are tokens 1A..12D. Control = LacZ, Resistant = ARoe."""
    control, resistant = [], []
    for token in matrix.columns:
        vector = MAXQUANT_SAMPLE[_token_number(token)][1]
        (resistant if vector == "ARoe" else control).append(token)
    return control, resistant


def maxquant_labels(matrix: pd.DataFrame):
    """Per-column label vector: 1 = resistant (ARoe), 0 = control (LacZ)."""
    return [
        1 if MAXQUANT_SAMPLE[_token_number(t)][1] == "ARoe" else 0
        for t in matrix.columns
    ]


def maxquant_sample_label(token: str) -> str:
    """Human-readable label for a sample token, e.g. '5A' -> 'M14·ARoe·Dab (A)'."""
    num = _token_number(token)
    rep = _TOKEN_RE.match(token).group(2)
    cell, vector, treat = MAXQUANT_SAMPLE[num]
    treat_short = "Dab" if treat == "Dabrafenib" else "DMSO"
    return f"{cell}·{vector}·{treat_short} ({rep})"


def maxquant_cell_line(token: str) -> str:
    return MAXQUANT_SAMPLE[_token_number(token)][0]


# ──────────────────────────────────────────────────────────────────────────
# Experimental design — Transcriptomics (GEO GSE199405) — REAL METADATA
# ──────────────────────────────────────────────────────────────────────────
# GSM -> (cell line, vector, treatment)  (extracted from !Sample_title)
GEO_SAMPLE = {
    "GSM5972234": ("A375", "ARoe", "Dabrafenib"),
    "GSM5972235": ("A375", "ARoe", "DMSO"),
    "GSM5972236": ("A375", "LacZ", "Dabrafenib"),
    "GSM5972237": ("A375", "LacZ", "DMSO"),
    "GSM5972238": ("M14", "ARoe", "Dabrafenib"),
    "GSM5972239": ("M14", "ARoe", "DMSO"),
    "GSM5972240": ("M14", "LacZ", "Dabrafenib"),
    "GSM5972241": ("M14", "LacZ", "DMSO"),
    "GSM5972242": ("WM9", "ARoe", "Dabrafenib"),
    "GSM5972243": ("WM9", "ARoe", "DMSO"),
    "GSM5972244": ("WM9", "LacZ", "Dabrafenib"),
    "GSM5972245": ("WM9", "LacZ", "DMSO"),
}


def geo_groups(matrix: pd.DataFrame):
    """Return (control_columns, resistant_columns) of the GEO matrix.
    Control = LacZ, Resistant = ARoe."""
    control, resistant = [], []
    for col in matrix.columns:
        info = GEO_SAMPLE.get(str(col))
        if info is None:
            continue
        (resistant if info[1] == "ARoe" else control).append(col)
    return control, resistant


# ──────────────────────────────────────────────────────────────────────────
# Missing-value imputation (Perseus-style, standard in phosphoproteomics)
# ──────────────────────────────────────────────────────────────────────────
def min_value_impute(log2_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values of a log2 matrix.

    In MS data, intensity 0 usually means "not detected", not random noise.
    Each 0 is treated as missing and filled with a low down-shifted value
    (mean − 1.8·std of the detected values in the column), a convention
    popularised by the Perseus software.
    """
    out = log2_df.copy()
    for col in out.columns:
        detected = out[col].replace(0, np.nan)
        vals = detected.dropna()
        fill = (vals.mean() - 1.8 * vals.std()) if len(vals) > 1 else 0.0
        out[col] = detected.fillna(fill)
    return out


# Gene sets per signalling pathway (for pathway-activity inference) ────────────
PATHWAY_GENES = {
    "MAPK/ERK": {
        "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "BRAF", "RAF1", "ARAF",
        "RPS6KA1", "RPS6KA3", "MAPK7", "DUSP6", "EGFR", "SOS1", "KRAS", "NRAS",
    },
    "PI3K-AKT": {
        "AKT1", "AKT2", "AKT3", "PIK3CA", "PIK3CB", "PIK3R1", "PDPK1",
        "GSK3B", "GSK3A", "PTEN", "FOXO1", "FOXO3", "TSC2",
    },
    "mTOR": {
        "MTOR", "RPTOR", "RICTOR", "RPS6KB1", "RPS6", "EIF4EBP1", "EIF4E",
        "AKT1S1", "ULK1",
    },
    "EMT": {
        "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "VIM", "CDH2", "FN1",
        "CTNNB1", "AXL", "WNT5A",
    },
}
