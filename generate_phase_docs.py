# -*- coding: utf-8 -*-
"""
generate_phase_docs.py
==============================================================================
Generates the phase-organized documentation of the melanoma multi-omics
pipeline into docs/phases/.

The project can be read two complementary ways:

  • Technical order    — the 21 executable stages, in the order run.py runs them.
  • Biological phases  — the signaling-transition-kinetics model of combined
                         BRAFi + MEKi treatment (see docs/phase2/): three phases
                         that unfold over time, from acute MAPK shutdown to
                         consolidated resistance.

This script reconciles the two: it reads the curated per-script catalog
(docs/phases/_pipeline_metadata.json), the run order (run.py), and the actual
files on disk (results/figures, results/outputs, results/logs), then writes:

  docs/phases/README.md                  — index + both views + dataset table
  docs/phases/00_foundation.md           — data import, cleaning, QC
  docs/phases/01_acute_suppression.md    — Phase 1 (6 h – 24 h)
  docs/phases/02_adaptive_rewiring.md    — Phase 2 (48 h – 72 h)
  docs/phases/03_consolidated_resistance.md  — Phase 3 (3 d – 90 d)

Run:  python3 generate_phase_docs.py
==============================================================================
"""
import ast
import json
import os
from collections import defaultdict

DOCS_DIR = "docs/phases"
META_FILE = os.path.join(DOCS_DIR, "_pipeline_metadata.json")
FIG_DIR = "results/figures"
OUT_DIR = "results/outputs"
LOG_DIR = "results/logs"
RUN_FILE = "run.py"

# ---------------------------------------------------------------------------
# Biological-phase narrative (from the signaling-transition-kinetics model,
# docs/phase2/). Order matters — it drives the document order.
# ---------------------------------------------------------------------------
PHASES = [
    ("foundation", {
        "file": "00_foundation.md",
        "title": "Foundation — Data Import, Cleaning & Quality Control",
        "timeframe": "—",
        "summary":
            "Before any biological interpretation, the pipeline loads the raw "
            "multi-omics datasets, removes contaminants and low-confidence "
            "measurements, log2-normalises and imputes missing values, and "
            "confirms that samples are comparable. These stages produce the "
            "analysis-ready matrices every later phase consumes.",
    }),
    ("phase1", {
        "file": "01_acute_suppression.md",
        "title": "Phase 1 — Acute Suppression (6 h – 24 h)",
        "timeframe": "6 h – 24 h",
        "summary":
            "Immediate therapeutic efficacy. Combined BRAFi + MEKi shut down the "
            "MAPK pathway: KSEA scores on target phosphosites and ERK1/2 "
            "activation drop sharply, confirming target blockade. The cell is "
            "driven into full pathway suppression. This phase is anchored by the "
            "acute 30-minute inhibitor phosphoproteome (PXD013923) and the "
            "kinase-substrate enrichment analysis (KSEA / OmniPath).",
    }),
    ("phase2", {
        "file": "02_adaptive_rewiring.md",
        "title": "Phase 2 — Adaptive Rewiring (48 h – 72 h)",
        "timeframe": "48 h – 72 h",
        "summary":
            "Emergency adaptation. Loss of MAPK negative feedback triggers the "
            "emergency activation of parallel routes: PI3K/AKT/mTOR pathway "
            "scores and receptor tyrosine-kinase expression (AXL, PDGFR) rebound "
            "as kinase activities recover. The cell begins to bypass the block. "
            "This phase covers the temporal kinase-activation dynamics, the "
            "pathway-activity readout, and the rewired signaling network.",
    }),
    ("phase3", {
        "file": "03_consolidated_resistance.md",
        "title": "Phase 3 — Consolidated Resistance (3 d – 90 d)",
        "timeframe": "3 d – 90 d",
        "summary":
            "Stable resistance and a permanent evolutionary shift toward fully "
            "integrated, alternative signaling. Multi-omics heatmaps show the "
            "phosphoproteomic rebound; long-term RNA-seq (GSE110054) reveals "
            "stable, concordant gene-expression signatures; phenotypic switching "
            "yields invasive cells. This phase gathers the stable resistant-vs-"
            "control signatures, cross-genotype baselines, the clinical TCGA-SKCM "
            "cohort, multi-omics integration, and the predictive models.",
    }),
]
PHASE_ORDER = [k for k, _ in PHASES]
PHASE_INFO = dict(PHASES)

# Datasets used across the project (for the README table).
DATASETS = [
    ("PXD013923", "Phosphoproteome (SILAC)", "Acute BRAFi/MEKi/ERKi in A375, 30 min", "Phase 1"),
    ("OmniPath", "Kinase-substrate network", "39,037 enzyme→site relations for KSEA", "Phase 1"),
    ("PXD022992", "Phosphoproteome (directDIA)", "6 melanoma cell lines, basal", "Phase 3"),
    ("GSE110054", "Transcriptome (RNA-seq)", "BRAFi time course, M229/M397 (3 d–90 d)", "Phase 3"),
    ("TCGA-SKCM", "Genomics + clinical", "470 cutaneous-melanoma patients", "Phase 3"),
    ("PXD026557 / GSE199405", "Phospho + protein + microarray", "AR-overexpression (resistant) vs control", "Foundation / Phase 3"),
]

GENERATED_NOTE = "> _Auto-generated by `generate_phase_docs.py` — do not edit by hand; re-run the generator to update._"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_meta():
    with open(META_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def parse_run_order():
    """Return dict script -> (pipeline_number, label) parsed from run.py."""
    with open(RUN_FILE, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    order = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "PIPELINE":
                    for elt in node.value.elts:
                        num, label, script = [
                            e.value if isinstance(e, ast.Constant) else None
                            for e in elt.elts
                        ]
                        order[script] = (num, label)
    return order


def exists(kind, name):
    d = FIG_DIR if kind == "fig" else OUT_DIR
    return os.path.exists(os.path.join(d, name))


def has_log(script):
    base = os.path.splitext(script)[0]
    return os.path.exists(os.path.join(LOG_DIR, base + ".log"))


def md_list(items):
    return "\n".join(f"- {it}" for it in items) if items else "_none_"


# ---------------------------------------------------------------------------
# Per-script rendering
# ---------------------------------------------------------------------------
def render_script(script, m, run_order, idx):
    num, label = run_order.get(script, (None, None))
    lines = []
    title = m.get("title") or label or script
    lines.append(f"### {idx}. {title}")
    lines.append("")
    lines.append(f"`{script}`" + (f" · pipeline stage **{num}**" if num is not None else ""))
    lines.append("")
    if m.get("objective"):
        lines.append(m["objective"])
        lines.append("")

    if m.get("datasets"):
        lines.append("**Datasets / inputs**")
        lines.append(md_list(m["datasets"]))
        lines.append("")

    # figures / tables verified against disk
    figs = []
    for f in m.get("figures", []):
        figs.append(f"`{f}`" + ("" if exists("fig", f) else "  _(not found)_"))
    tbls = []
    for t in m.get("tables", []):
        tbls.append(f"`{t}`" + ("" if exists("out", t) else "  _(not found)_"))

    if figs:
        lines.append("**Figures** → `results/figures/`")
        lines.append(md_list(figs))
        lines.append("")
    if tbls:
        lines.append("**Tables** → `results/outputs/`")
        lines.append(md_list(tbls))
        lines.append("")

    if m.get("highlights"):
        lines.append("**Key results**")
        lines.append(md_list(m["highlights"]))
        lines.append("")

    run_cmd = m.get("run") or f"python3 {script}"
    lines.append(f"**Run:** `{run_cmd}`"
                 + ("  · log captured" if has_log(script) else ""))
    lines.append("")

    if m.get("notes"):
        lines.append(f"> **Note:** {m['notes']}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase documents
# ---------------------------------------------------------------------------
def write_phase_doc(phase_key, meta, run_order):
    info = PHASE_INFO[phase_key]
    scripts = [s for s, m in meta.items() if m["phase"] == phase_key]
    # order scripts by their run.py pipeline number (fallback: name)
    scripts.sort(key=lambda s: (run_order.get(s, (999, ""))[0] if run_order.get(s) else 999, s))

    nums = sorted({run_order[s][0] for s in scripts if s in run_order})
    nums_str = " ".join(str(n) for n in nums)

    lines = [f"# {info['title']}", "", GENERATED_NOTE, ""]
    if info["timeframe"] != "—":
        lines.append(f"**Timeframe:** {info['timeframe']}")
        lines.append("")
    lines.append(info["summary"])
    lines.append("")
    lines.append(f"**Stages in this phase:** {len(scripts)}")
    lines.append("")
    if nums:
        lines.append("Run every stage of this phase:")
        lines.append("")
        lines.append("```bash")
        lines.append(f"python3 run.py {nums_str}")
        lines.append("```")
        lines.append("")
    lines.append("[← Back to index](README.md)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, s in enumerate(scripts, 1):
        lines.append(render_script(s, meta[s], run_order, i))
        lines.append("---")
        lines.append("")

    path = os.path.join(DOCS_DIR, info["file"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path, len(scripts)


# ---------------------------------------------------------------------------
# Output inventory / consistency
# ---------------------------------------------------------------------------
def inventory(meta):
    referenced_figs, referenced_tbls = set(), set()
    missing = []
    for s, m in meta.items():
        for f in m.get("figures", []):
            referenced_figs.add(f)
            if not exists("fig", f):
                missing.append(f"`{f}` (figure, referenced by {s})")
        for t in m.get("tables", []):
            referenced_tbls.add(t)
            if not exists("out", t):
                missing.append(f"`{t}` (table, referenced by {s})")

    on_disk_figs = set(os.listdir(FIG_DIR)) if os.path.isdir(FIG_DIR) else set()
    on_disk_tbls = set(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else set()
    orphan_figs = sorted(f for f in on_disk_figs if f.endswith(".png") and f not in referenced_figs)
    orphan_tbls = sorted(t for t in on_disk_tbls
                         if t.endswith((".csv", ".graphml")) and t not in referenced_tbls)
    return {
        "n_fig_disk": len([f for f in on_disk_figs if f.endswith('.png')]),
        "n_tbl_disk": len([t for t in on_disk_tbls if t.endswith(('.csv', '.graphml'))]),
        "missing": missing,
        "orphan_figs": orphan_figs,
        "orphan_tbls": orphan_tbls,
    }


# ---------------------------------------------------------------------------
# README / index
# ---------------------------------------------------------------------------
def write_readme(meta, run_order, phase_counts, inv):
    L = []
    L.append("# Melanoma Multi-omics Pipeline — Phase Documentation")
    L.append("")
    L.append(GENERATED_NOTE)
    L.append("")
    L.append(
        "This pipeline studies how melanoma cells respond and become resistant to "
        "combined BRAF + MEK inhibition (BRAFi + MEKi). It integrates targeted and "
        "data-independent phosphoproteomics, transcriptomics, clinical genomics and "
        "machine learning across the full time course of the resistance process.")
    L.append("")

    # --- biological phases ---
    L.append("## The signaling-transition-kinetics model")
    L.append("")
    L.append(
        "The project is organized around three biological phases of combined "
        "BRAFi + MEKi treatment (see `docs/phase2/`), plus a foundation layer:")
    L.append("")
    L.append("| Phase | Timeframe | What happens | Document |")
    L.append("|-------|-----------|--------------|----------|")
    label_short = {
        "foundation": ("Foundation", "Data import, cleaning, QC"),
        "phase1": ("**Phase 1** — Acute suppression", "MAPK shutdown; KSEA & ERK1/2 drop"),
        "phase2": ("**Phase 2** — Adaptive rewiring", "PI3K/AKT/mTOR & RTK rebound; feedback loss"),
        "phase3": ("**Phase 3** — Consolidated resistance", "Stable multi-omics signatures; invasive switch"),
    }
    for k in PHASE_ORDER:
        info = PHASE_INFO[k]
        short, what = label_short[k]
        tf = info["timeframe"]
        L.append(f"| {short} | {tf} | {what} | [{info['file']}]({info['file']}) |")
    L.append("")

    # --- datasets ---
    L.append("## Datasets")
    L.append("")
    L.append("| Accession | Type | Content | Phase |")
    L.append("|-----------|------|---------|-------|")
    for acc, typ, content, phase in DATASETS:
        L.append(f"| {acc} | {typ} | {content} | {phase} |")
    L.append("")

    # --- reconciliation map ---
    L.append("## Biological phase ↔ pipeline stage")
    L.append("")
    L.append(
        "The executable stages (`run.py`) map onto the biological phases as "
        "follows. A single biological phase can span several pipeline stages.")
    L.append("")
    L.append("| Stage | Script | Biological phase |")
    L.append("|:-----:|--------|------------------|")
    phase_label = {"foundation": "Foundation", "phase1": "Phase 1", "phase2": "Phase 2", "phase3": "Phase 3"}
    rows = []
    for s, m in meta.items():
        num, _ = run_order.get(s, (None, None))
        rows.append((num if num is not None else 999, s, phase_label[m["phase"]]))
    for num, s, ph in sorted(rows):
        num_disp = str(num) if num != 999 else "—"
        rows_num = num_disp
        L.append(f"| {rows_num} | `{s}` | {ph} |")
    L.append("")

    # --- how to run ---
    L.append("## How to run")
    L.append("")
    L.append("```bash")
    L.append("pip install -r req.txt")
    L.append("python3 run.py            # run every stage in order")
    L.append("python3 run.py 15 16 17   # run only selected stages")
    L.append("python3 generate_phase_docs.py   # regenerate this documentation")
    L.append("```")
    L.append("")
    L.append(
        "Every stage writes its figures to `results/figures/`, tables to "
        "`results/outputs/`, and a console log to `results/logs/`.")
    L.append("")

    # --- inventory ---
    L.append("## Output inventory")
    L.append("")
    L.append(f"- **Figures on disk:** {inv['n_fig_disk']}  (`results/figures/`)")
    L.append(f"- **Tables on disk:** {inv['n_tbl_disk']}  (`results/outputs/`)")
    L.append(f"- **Documented stages:** {sum(phase_counts.values())}")
    L.append("")
    if inv["missing"]:
        L.append("**Referenced but not found (run the stage to produce it):**")
        L.append(md_list(inv["missing"]))
        L.append("")
    L.append("")
    L.append(
        "_The full narrative report (all datasets, methodology and results) is in "
        "`Relatorio_Completo_Melanoma_MultiOmica.pdf`, generated by "
        "`generate_full_report_en.py`._")
    L.append("")

    path = os.path.join(DOCS_DIR, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    meta = load_meta()
    run_order = parse_run_order()

    print("Generating phase documentation into docs/phases/ ...")
    phase_counts = {}
    for k in PHASE_ORDER:
        path, n = write_phase_doc(k, meta, run_order)
        phase_counts[k] = n
        print(f"  [{k:11s}] {n:2d} stages  ->  {path}")

    inv = inventory(meta)
    readme = write_readme(meta, run_order, phase_counts, inv)
    print(f"  [index      ]           ->  {readme}")

    print(f"\nFigures on disk: {inv['n_fig_disk']} | Tables on disk: {inv['n_tbl_disk']}")
    if inv["missing"]:
        print(f"Referenced-but-missing: {len(inv['missing'])}")
    if inv["orphan_figs"] or inv["orphan_tbls"]:
        print(f"Unreferenced on disk: {len(inv['orphan_figs'])} figures, "
              f"{len(inv['orphan_tbls'])} tables (likely produced by run.py stages "
              f"1–14 whose outputs are catalogued in results/SUMMARY.md)")
    print("Done.")


if __name__ == "__main__":
    main()
