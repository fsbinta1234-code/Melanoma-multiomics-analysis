# Pipeline results — core run (stages 1–14)

This file summarizes the **core resistant-vs-control run** (pipeline stages
1–14, the AR-overexpression dataset). The later real-data stages (15–21:
PXD013923, PXD022992, KSEA, TCGA-SKCM, GSE110054, neural network) are documented
per phase in [`../docs/phases/`](../docs/phases/README.md), which is the
authoritative, phase-organized documentation for the whole project.

Figures are in [figures/](figures/), tables in [outputs/](outputs/) and logs in
[logs/](logs/).

- **Environment:** conda `Melanoma`, Python 3.11.15, `scikit-learn 1.9.0`, `networkx 3.6.1`
- **Compared groups:** **ARoe (resistant)** vs **LacZ (control)** — AR
  overexpression confers BRAFi/MEKi resistance (see
  [`../docs/phases/README.md`](../docs/phases/README.md)).

## What was fixed

| Original problem | Fix |
|------------------|-----|
| Selection grabbed aggregate columns (`Intensity`, `Intensity___1`…) | Robust selection of the real sample columns `1A`–`12D` only (48 samples = 12 conditions × 4 replicates) |
| Meaningless `0:3 / 3:6` grouping → 0 significant → heatmap crash | Real ARoe vs LacZ groups (24 vs 24) → 714 significant sites; heatmap works |
| ML: `inconsistent samples [204, 6]` | 48 samples with labels derived from the experimental design |
| PCA: `IndexError` (6 colours for 204 points) | Labels/colours generated programmatically for the 48 samples |
| GEO re-logged (double log2) | Uses RMA log2 values directly; ARoe/LacZ groups from real metadata |
| Incomplete `req.txt` | Added `scikit-learn` and `networkx` |
| Figures not saved / `plt.show()` blocked | Consistent visual theme + automatic saving to `results/figures/` |

## Key findings

- **Differential phosphorylation:** **714 significant** sites (FDR<0.05, |log2FC|>1) —
  **704 ↑ in resistant**, 10 ↓. Top hits: `NBN_S397`, `MKI67_S1679`, `NAV1_S391`,
  `BAD_S118` (an AKT substrate), `PEX19_S147`.
- **Pathway activity** (mean log2FC of pathway sites, all ↑ in resistant):
  MAPK/ERK 0.42 · EMT 0.38 · mTOR 0.37 · PI3K-AKT 0.26 — consistent with adaptive
  signalling reactivation in resistance.
- **Proteomics:** 0 significant proteins; **Transcriptomics:** 1 probe. The
  heterogeneity across the 3 cell lines (A375/M14/WM9) dominates the variance
  when pooled, diluting the AR effect on the steady-state proteome/transcriptome —
  while phosphorylation (signalling) captures the effect clearly.
- **Multi-omics integration:** 3059 shared genes (phospho ∩ protein),
  log2FC correlation r = 0.137 (weak positive concordance).
- **Machine Learning (Random Forest):** accuracy **5-fold CV = 0.836 ± 0.104**
  (held-out test 0.60 — small-sample variance).
- **PCA:** PC1 = 39.9%, PC2 = 8.0% of the variance.
- **Signalling network:** 8 nodes, 10 edges → `Melanoma_Resistance_Network.graphml`.

## Figures (16)

| File | Phase | Content |
|------|:-----:|---------|
| `02_quality_control_boxplots.png` | 2 | Intensity distributions (3 layers) |
| `03_temporal_kinase_activation.png` | 3 | Temporal curves (illustrative) |
| `04_differential_phosphorylation_volcano.png` | 4 | Gene-annotated volcano |
| `04_differential_phosphorylation_heatmap.png` | 4 | Clustermap of the top 40 sites |
| `04_phospho_group_comparison_top_sites.png` | 4 | Top 20 sites by \|log2FC\| |
| `05_volcano_plot.png` | 5 | Volcano (publication) |
| `06_pathway_activity_scores.png` | 6 | Per-pathway activity |
| `07_proteomics_volcano.png` / `_fc_distribution.png` | 7 | Proteomics |
| `08_transcriptomics_volcano.png` / `_fc_distribution.png` | 8 | Transcriptomics |
| `09_multiomics_phospho_vs_protein.png` | 9 | Phospho × proteome per gene |
| `10_ml_confusion_matrix.png` / `_feature_importance.png` | 10 | ML |
| `11_pca_clustering.png` | 11 | PCA + K-Means |
| `12_resistance_signaling_network.png` | 12 | Signalling network |

## Tables (outputs/)

`Differential_Phosphorylation_all.csv` (15124) · `_significant.csv` (714) ·
`Proteomics_Results.csv` (6435) · `Transcriptomics_Results.csv` (135750) ·
`Integrated_Multiomics.csv` (3059) · `Pathway_Activity_Scores.csv` ·
`Melanoma_Resistance_Network.graphml`

## Reproduce

```bash
pip install -r req.txt           # already includes scikit-learn and networkx
python3 final_result_export.py   # core (phases 4→7→8→9) + CSVs
# or run each phase individually — every script saves its own figures/tables
```

> **Assumption to verify:** the MaxQuant 1–12 sample → condition mapping assumes
> the same order as GEO. The transcriptomics layer uses the real metadata mapping.
> See `pipeline_config.py` (`MAXQUANT_SAMPLE` dictionary).
