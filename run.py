#!/usr/bin/env python3
"""
run.py — Run the full melanoma multi-omics pipeline (all 14 phases).

Each phase is executed as an isolated subprocess (so one failure does not stop
the others), with its console output captured to results/logs/<phase>.log.
Every script saves its own figures to results/figures/ and tables to
results/outputs/. A summary table is printed at the end.

Usage
-----
    python run.py                 # run all phases in order
    python run.py 4 7 11          # run only the given phase numbers

`codeFatima.py` is intentionally excluded (not part of the pipeline — see
DOCUMENTATION.md §7).
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "results", "logs")

# (phase number, label, script). Several scripts share a phase number.
PIPELINE = [
    (1, "Data import", "readDataset.py"),
    (2, "Cleaning & QC", "cleanDatas.py"),
    (2, "Quality control", "quality_control_visualization.py"),
    (3, "Temporal dynamics", "temporal_kinase_activation_dynamics.py"),
    (4, "Differential phosphorylation", "differential_phosphorylation_analysis.py"),
    (4, "Group comparison", "phospho_group_comparison.py"),
    (5, "Volcano plot", "volcano_plot_visualization.py"),
    (6, "Pathway activity", "kinase_signaling_analysis.py"),
    (7, "Proteomics", "proteomics_analysis.py"),
    (8, "Transcriptomics", "transcriptomics_validation.py"),
    (9, "Multi-omics integration", "multi_omics_integration.py"),
    (10, "Machine learning", "machine_learning_resistance_prediction.py"),
    (11, "PCA & clustering", "pca_and_clustering_analysis.py"),
    (12, "Signalling network", "network_based_systems_biology.py"),
    (13, "Cytoscape export", "cytoscape_export.py"),
    (14, "Final export", "final_result_export.py"),
]


def run_script(label: str, script: str) -> tuple:
    """Run one script as a subprocess; return (ok, seconds, log_path)."""
    log_path = os.path.join(LOG_DIR, os.path.splitext(script)[0] + ".log")
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    with open(log_path, "w") as fh:
        fh.write(result.stdout)
        if result.stderr:
            fh.write("\n--- STDERR ---\n")
            fh.write(result.stderr)
    return result.returncode == 0, elapsed, log_path


def main(argv) -> int:
    os.makedirs(LOG_DIR, exist_ok=True)

    wanted = {int(a) for a in argv if a.isdigit()}
    phases = [p for p in PIPELINE if not wanted or p[0] in wanted]

    print("=" * 64)
    print("  Melanoma multi-omics pipeline")
    print(f"  {len(phases)} script(s) → results/  (figures, outputs, logs)")
    print("=" * 64)

    summary = []
    for number, label, script in phases:
        print(f"  [Phase {number:>2}] {label:<28} running…", flush=True)
        ok, elapsed, log_path = run_script(label, script)
        status = "OK  " if ok else "FAIL"
        print(f"            {status}  ({elapsed:5.1f}s)  → {os.path.relpath(log_path, BASE_DIR)}")
        if not ok:
            tail = _read_tail(log_path, 6)
            print("            last log lines:")
            for line in tail:
                print(f"              {line}")
        summary.append((number, label, ok, elapsed))

    print("\n" + "=" * 64)
    n_ok = sum(1 for *_, ok, _ in summary if ok)
    print(f"  SUMMARY: {n_ok}/{len(summary)} OK")
    for number, label, ok, elapsed in summary:
        print(f"    Phase {number:>2}  {'OK  ' if ok else 'FAIL'}  {elapsed:5.1f}s  {label}")
    print("=" * 64)
    print("  Figures : results/figures/")
    print("  Tables  : results/outputs/")
    print("  Logs    : results/logs/")

    return 0 if n_ok == len(summary) else 1


def _read_tail(path: str, n: int) -> list:
    try:
        with open(path) as fh:
            return [ln.rstrip() for ln in fh.readlines()[-n:]]
    except OSError:
        return []


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
