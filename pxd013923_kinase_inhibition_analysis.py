"""
Real-data analysis — PXD013923 (acute kinase-inhibition phosphoproteomics).

Dataset: "Global view of the RAF-MEK-ERK module..." (PRIDE PXD013923).
A375 melanoma cells, 30-min treatment with BRAFi (RAFi/Dabrafenib),
MEKi (Trametinib) or ERKi (SCH772984); MaxQuant SILAC. Protein IDs are Ensembl
(ENSP), mapped to gene symbols via datas/PXD013923/ensp_to_gene.csv (mygene.info).

We use the per-experiment `Ratio H/L normalized` columns (drug vs control, log2)
and quantify how each inhibitor reshapes the phosphoproteome:
  1. global response distribution per inhibitor;
  2. agreement between the three inhibitors (linear RAF->MEK->ERK module);
  3. MAPK/ERK pathway anchor — canonical pathway phosphosites are suppressed.

SILAC sign is anchored to biology: inhibiting the MAPK module must REDUCE
phosphorylation of canonical pathway sites, so the convention is set such that
negative log2 = suppressed by the inhibitor.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import pipeline_config as cfg

PHOSPHO_FILE = "datas/PXD013923/PhosphoSTYSites.txt"
ENSP_MAP_FILE = "datas/PXD013923/ensp_to_gene.csv"
LOC_CUTOFF = 0.75

# Inhibitor -> replicate experiment labels (from summary.txt; 30-min treatment)
INHIBITORS = {
    "BRAFi": ["RAF1", "RAF3", "RAF4"],
    "MEKi": ["MEK1", "MEK3", "MEK4"],
    "ERKi": ["ERK1", "ERK1b", "ERK3", "ERK5"],
}
INHIBITOR_COLORS = {"BRAFi": cfg.COLOR_RESIST, "MEKi": cfg.COLOR_CONTROL, "ERKi": cfg.COLOR_ACCENT}

# MAPK/ERK module + canonical ERK substrates (anchor set for sign + biology check)
ANCHOR_GENES = {
    "MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "BRAF", "RAF1", "ARAF",
    "RPS6KA1", "RPS6KA2", "RPS6KA3", "RPS6KA4", "RPS6KA5", "RPS6KB1", "RPS6KB2",
    "RPS6", "EIF4EBP1", "ELK1", "JUN", "FOS", "FOSL1", "ETV4", "ETV5",
    "DUSP4", "DUSP6", "SPRY2", "SPRY4", "GAB1", "GAB2", "SOS1", "EGFR", "BAD",
    "MKNK1", "MKNK2", "MAPKAPK2", "KSR1", "SHC1", "FRS2", "STMN1",
}


def _load_ensp_map():
    if not os.path.exists(ENSP_MAP_FILE):
        print(f"[warn] {ENSP_MAP_FILE} ausente — genes ficarão como ENSP.")
        return {}
    m = pd.read_csv(ENSP_MAP_FILE)
    return dict(zip(m["ensp"], m["gene"]))


def _make_unique(labels):
    seen, out = {}, []
    for x in labels:
        if x in seen:
            seen[x] += 1; out.append(f"{x}.{seen[x]}")
        else:
            seen[x] = 0; out.append(x)
    return pd.Index(out)


def load_inhibitor_matrix():
    """Return (matrix, meta): per-site mean log2 ratio per inhibitor, sign-anchored."""
    ensp2gene = _load_ensp_map()
    df = pd.read_csv(PHOSPHO_FILE, sep="\t", low_memory=False)
    for flag in ("Reverse", "Potential contaminant"):
        if flag in df.columns:
            df = df[df[flag] != "+"]
    df = df[df["Localization prob"] >= LOC_CUTOFF].reset_index(drop=True)

    ensp = df["Leading proteins"].astype(str).str.split(";").str[0].str.replace("_reference", "", regex=False)
    df = df[ensp.str.startswith("ENSP")].reset_index(drop=True)
    ensp = ensp[ensp.str.startswith("ENSP")].reset_index(drop=True)

    gene = ensp.map(ensp2gene).fillna(ensp)
    aa = df.get("Amino acid", pd.Series("?", index=df.index)).astype(str)
    pos = pd.to_numeric(df.get("Position"), errors="coerce").fillna(0).astype(int)
    site_id = _make_unique((gene + "_" + aa + pos.astype(str)).values)

    cols = {}
    for inh, exps in INHIBITORS.items():
        rcols = [f"Ratio H/L normalized {e}" for e in exps if f"Ratio H/L normalized {e}" in df.columns]
        vals = df[rcols].apply(pd.to_numeric, errors="coerce")
        cols[inh] = np.log2(vals.where(vals > 0)).mean(axis=1).values
    matrix = pd.DataFrame(cols, index=site_id)
    meta = pd.DataFrame({"gene": gene.values,
                         "is_mapk_anchor": gene.str.upper().isin(ANCHOR_GENES).values},
                        index=site_id)

    matrix = matrix.dropna(how="all")
    meta = meta.loc[matrix.index]

    anchor_vals = matrix.loc[meta["is_mapk_anchor"]].stack()
    anchor_median = anchor_vals.median() if len(anchor_vals) else 0.0
    if anchor_median > 0:
        matrix = -matrix
        anchor_median = -anchor_median
        print("[info] sinal invertido para que inibição da via MAPK fique negativa.")
    print(f"[info] sítios={len(matrix)} | sítios-âncora MAPK={int(meta['is_mapk_anchor'].sum())} "
          f"| mediana log2 da âncora={anchor_median:.3f}")
    return matrix, meta


def run():
    cfg.apply_style()
    matrix, meta = load_inhibitor_matrix()

    for inh in INHIBITORS:
        v = matrix[inh].dropna()
        print(f"  {inh:6s}: n={len(v):5d} | mediana log2={v.median():+.3f} | "
              f"down(<-1)={int((v < -1).sum())} up(>1)={int((v > 1).sum())}")

    _plot_distributions(matrix)
    _plot_inhibitor_agreement(matrix)
    _plot_mapk_anchor(matrix, meta)
    _export_tables(matrix, meta)


def _plot_distributions(matrix):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [matrix[i].dropna().values for i in INHIBITORS]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, inh in zip(parts["bodies"], INHIBITORS):
        pc.set_facecolor(INHIBITOR_COLORS[inh]); pc.set_alpha(0.7); pc.set_edgecolor("#333")
    ax.set_xticks(range(1, len(INHIBITORS) + 1)); ax.set_xticklabels(list(INHIBITORS))
    ax.axhline(0, ls="--", color="dimgrey", lw=0.8)
    ax.set_ylabel("log2 fold change (inhibitor / control)")
    ax.set_title("Acute phosphoproteome response to MAPK-pathway inhibition (A375, 30 min)")
    ax.set_ylim(-4, 4)
    cfg.save_figure(fig, "pxd013923_response_distributions")


def _plot_inhibitor_agreement(matrix):
    pairs = list(combinations(INHIBITORS, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.8))
    for ax, (a, b) in zip(axes, pairs):
        d = matrix[[a, b]].dropna()
        r = stats.pearsonr(d[a], d[b])[0] if len(d) > 2 else float("nan")
        ax.scatter(d[a], d[b], s=8, alpha=0.3, c=cfg.COLOR_NS, edgecolors="none")
        ax.axhline(0, color="dimgrey", lw=0.6); ax.axvline(0, color="dimgrey", lw=0.6)
        lim = 4
        ax.plot([-lim, lim], [-lim, lim], ls=":", color=cfg.COLOR_RESIST, lw=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"{a} (log2)"); ax.set_ylabel(f"{b} (log2)")
        ax.set_title(f"{a} vs {b}   r = {r:.2f}  (n={len(d)})", fontsize=11)
    fig.suptitle("Agreement between BRAFi / MEKi / ERKi responses (linear RAF→MEK→ERK module)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    cfg.save_figure(fig, "pxd013923_inhibitor_agreement")


def _plot_mapk_anchor(matrix, meta):
    rows = []
    for inh in INHIBITORS:
        for v in matrix.loc[meta["is_mapk_anchor"], inh].dropna():
            rows.append((inh, "MAPK pathway", v))
        for v in matrix.loc[~meta["is_mapk_anchor"], inh].dropna():
            rows.append((inh, "Other sites", v))
    long = pd.DataFrame(rows, columns=["Inhibitor", "Set", "log2"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=long, x="Inhibitor", y="log2", hue="Set", showfliers=False,
                palette={"MAPK pathway": cfg.COLOR_RESIST, "Other sites": "#cfd4da"}, ax=ax)
    ax.axhline(0, ls="--", color="dimgrey", lw=0.8)
    ax.set_ylabel("log2 fold change (inhibitor / control)")
    ax.set_title("Canonical MAPK/ERK phosphosites are selectively suppressed by inhibition")
    cfg.save_figure(fig, "pxd013923_mapk_anchor")


def _export_tables(matrix, meta):
    cfg.save_table(meta.join(matrix), "PXD013923_inhibitor_log2_matrix.csv")
    tops = []
    for inh in INHIBITORS:
        t = meta.join(matrix)[["gene", inh]].dropna(subset=[inh]).nsmallest(15, inh).copy()
        t.insert(0, "inhibitor", inh); t = t.rename(columns={inh: "log2FC"})
        tops.append(t)
    cfg.save_table(pd.concat(tops), "PXD013923_top_suppressed_sites.csv")


def main():
    run()


if __name__ == "__main__":
    main()
