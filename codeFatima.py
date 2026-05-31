# ==========================================================
# MELANOMA PHOSPHOPROTEOMICS ANALYSIS
# TEMPORAL KINASE ACTIVATION DYNAMICS
# BRAFi/MEKi RESISTANCE ANALYSIS
# ==========================================================

# ==========================================================
# STEP 1 — IMPORT REQUIRED LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# STEP 2 — LOAD PHOSPHOPROTEOMICS DATASET
# ==========================================================

# Load the phosphoproteomics dataset
# downloaded from PRIDE (PXD026557)

phospho = pd.read_csv(
    "datas/Phospho__STY_Sites.csv",
    sep="\t",
    low_memory=False
)

print(phospho.head())

# ==========================================================
# STEP 3 — REMOVE CONTAMINANTS
# ==========================================================

phospho_clean = phospho[
    (phospho["Reverse"] != "+") &
    (phospho["Potential contaminant"] != "+")
]

print(
    "Dataset after contaminant removal:",
    phospho_clean.shape
)

# ==========================================================
# STEP 4 — FILTER HIGH-CONFIDENCE PHOSPHOSITES
# ==========================================================

# Keep only phosphosites with localization probability >= 0.75

phospho_filtered = phospho_clean[
    phospho_clean["Localization prob"] >= 0.75
].reset_index(drop=True)

print(
    "Dataset after phosphosite filtering:",
    phospho_filtered.shape
)

# ==========================================================
# STEP 5 — IDENTIFY INTENSITY COLUMNS
# ==========================================================

intensity_columns = [
    column
    for column in phospho_filtered.columns
    if "Intensity" in column
]

print("Intensity Columns:")
print(intensity_columns)

# ==========================================================
# STEP 6 — CREATE INTENSITY MATRIX
# ==========================================================

intensity_matrix = phospho_filtered[intensity_columns].copy()

print(intensity_matrix.head())

# ==========================================================
# STEP 7 — LOG2 NORMALIZATION
# ==========================================================

# log2(Intensity + 1) stabilizes variance and reduces extreme values

log2_matrix = np.log2(intensity_matrix + 1)

print(log2_matrix.head())

# ==========================================================
# STEP 8 — HANDLE MISSING VALUES
# ==========================================================

# Min-value imputation: mean - 1.8 * std of detected values per column
# Standard approach for phosphoproteomics (Perseus convention)
# Missing values in MS data typically represent true absence, not random noise

def min_value_impute(df):
    result = df.copy()
    for col in df.columns:
        detected = df[col].replace(0, np.nan).dropna()
        if len(detected) > 0:
            fill_val = detected.mean() - 1.8 * detected.std()
        else:
            fill_val = 0
        result[col] = df[col].replace(0, np.nan).fillna(fill_val)
    return result

log2_matrix = min_value_impute(log2_matrix)

print("Missing values replaced successfully")

# ==========================================================
# STEP 9 — ADD GENE NAMES
# ==========================================================

# Use .values to avoid index misalignment after filtering/reset_index
log2_matrix["Gene"] = phospho_filtered["Gene names"].values

# ==========================================================
# STEP 10 — SELECT IMPORTANT KINASES
# ==========================================================

# MAPK pathway kinases involved in BRAFi/MEKi resistance

kinases_of_interest = [
    "MAPK1",   # ERK2
    "MAPK3",   # ERK1
    "MAP2K1",  # MEK1
    "MAP2K2",  # MEK2
    "AKT1",
    "MTOR"
]

kinase_data = log2_matrix[
    log2_matrix["Gene"].isin(kinases_of_interest)
]

print(kinase_data.head())

# ==========================================================
# STEP 11 — IDENTIFY TIME POINT COLUMNS
# ==========================================================

# Case-insensitive matching to handle different column naming conventions

intensity_columns_lower = {col: col.lower() for col in intensity_columns}

time_6h  = [col for col, low in intensity_columns_lower.items() if "6h"  in low]
time_24h = [col for col, low in intensity_columns_lower.items() if "24h" in low]
time_48h = [col for col, low in intensity_columns_lower.items() if "48h" in low]
time_72h = [col for col, low in intensity_columns_lower.items() if "72h" in low]

print("6h Columns:",  time_6h)
print("24h Columns:", time_24h)
print("48h Columns:", time_48h)
print("72h Columns:", time_72h)

# Validate that all time points were found
for label, cols in [("6h", time_6h), ("24h", time_24h), ("48h", time_48h), ("72h", time_72h)]:
    if not cols:
        raise ValueError(
            f"No intensity columns found for time point '{label}'. "
            f"Check that column names contain '{label}' (case-insensitive). "
            f"Available columns: {intensity_columns}"
        )

# ==========================================================
# STEPS 12–17 — CALCULATE ACTIVATION SCORES PER TIME POINT
# ==========================================================

def activation_score(data, cols):
    subset = data[cols]
    return subset.mean(axis=1)  # per-kinase mean across replicates

scores_6h  = activation_score(kinase_data, time_6h)
scores_24h = activation_score(kinase_data, time_24h)
scores_48h = activation_score(kinase_data, time_48h)
scores_72h = activation_score(kinase_data, time_72h)

activation_6h  = scores_6h.mean()
activation_24h = scores_24h.mean()
activation_48h = scores_48h.mean()
activation_72h = scores_72h.mean()

print("6h Activation Score:",  activation_6h)
print("24h Activation Score:", activation_24h)
print("48h Activation Score:", activation_48h)
print("72h Activation Score:", activation_72h)

# ==========================================================
# STEP 18 — CREATE TEMPORAL ACTIVATION TABLE
# ==========================================================

time_points = ["6h", "24h", "48h", "72h"]

activation_scores = [
    activation_6h,
    activation_24h,
    activation_48h,
    activation_72h
]

temporal_results = pd.DataFrame({
    "TimePoint":       time_points,
    "ActivationScore": activation_scores
})

print(temporal_results)

# ==========================================================
# STEP 19 — PER-KINASE TEMPORAL ACTIVATION TABLE
# ==========================================================

# Build a kinase × time-point matrix for richer interpretation

kinase_temporal = pd.DataFrame({
    "Gene": kinase_data["Gene"].values,
    "6h":   scores_6h.values,
    "24h":  scores_24h.values,
    "48h":  scores_48h.values,
    "72h":  scores_72h.values
})

print(kinase_temporal)

# ==========================================================
# STEP 20 — GENERATE PLOTS
# ==========================================================

# --- Plot 1: Overall temporal activation curve ---

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(
    temporal_results["TimePoint"],
    temporal_results["ActivationScore"],
    marker='o',
    linewidth=3
)
axes[0].set_xlabel("Treatment Time")
axes[0].set_ylabel("Mean Kinase Activation Score (log2)")
axes[0].set_title("Temporal Kinase Activation Dynamics\nDuring BRAFi/MEKi Resistance")
axes[0].grid(True)

# --- Plot 2: Per-kinase activation curves ---

for _, row in kinase_temporal.iterrows():
    axes[1].plot(
        ["6h", "24h", "48h", "72h"],
        [row["6h"], row["24h"], row["48h"], row["72h"]],
        marker='o',
        linewidth=2,
        label=row["Gene"]
    )

axes[1].set_xlabel("Treatment Time")
axes[1].set_ylabel("Activation Score (log2)")
axes[1].set_title("Per-Kinase Temporal Activation\nDuring BRAFi/MEKi Resistance")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("Temporal_Kinase_Activation.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# STEP 21 — EXPORT RESULTS
# ==========================================================

temporal_results.to_csv(
    "Temporal_Kinase_Activation.csv",
    index=False
)

kinase_temporal.to_csv(
    "Per_Kinase_Temporal_Activation.csv",
    index=False
)

print("Temporal kinase analysis completed successfully")
