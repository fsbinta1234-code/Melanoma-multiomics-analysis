"""
Download TCGA-SKCM data from the NCI GDC public API.

Downloads (open access, no authentication required):
  1. Clinical data  — cases × clinical variables (survival, stage, treatment)
  2. Somatic mutations — BRAF/NRAS/NF1/PTEN/CDKN2A/KIT driver mutations
     via the GDC /ssm_occurrences endpoint (no MAF download needed)
  3. RNA-seq manifest — gene counts for 50 representative samples
     (selected across mutation subtypes); saves file IDs + case IDs for
     downstream download if needed

All saved to datas/TCGA-SKCM/.
"""
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

OUT_DIR = "datas/TCGA-SKCM"
os.makedirs(OUT_DIR, exist_ok=True)

GDC_API = "https://api.gdc.cancer.gov"

DRIVER_GENES = ["BRAF", "NRAS", "NF1", "PTEN", "CDKN2A", "KIT", "MAP2K1",
                "RAC1", "PPP6C", "PREX2", "IDH1"]


# ---------------------------------------------------------------------------
# GDC helpers
# ---------------------------------------------------------------------------
def _gdc_post(endpoint: str, payload: dict, timeout: int = 60) -> dict:
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{GDC_API}/{endpoint}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"curl error: {r.stderr[:200]}")
    return json.loads(r.stdout)


def _gdc_get(endpoint: str, params: str = "", timeout: int = 60) -> dict:
    url = f"{GDC_API}/{endpoint}"
    if params:
        url += "?" + params
    cmd = ["curl", "-s", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return json.loads(r.stdout)


# ---------------------------------------------------------------------------
# 1. Clinical data
# ---------------------------------------------------------------------------
def download_clinical():
    print("[1] Downloading TCGA-SKCM clinical data...")
    payload = {
        "filters": {
            "op": "in",
            "content": {"field": "project.project_id", "value": ["TCGA-SKCM"]}
        },
        "fields": ",".join([
            "case_id", "submitter_id",
            "demographic.vital_status", "demographic.days_to_death",
            "demographic.days_to_birth",
            "diagnoses.days_to_last_follow_up",
            "diagnoses.tumor_stage", "diagnoses.ajcc_pathologic_stage",
            "diagnoses.prior_treatment",
            "exposures.pack_years_smoked",
            "summary.file_count",
        ]),
        "format": "JSON",
        "size": "500",
    }
    data = _gdc_post("cases", payload)
    hits = data["data"]["hits"]
    total = data["data"]["pagination"]["total"]
    print(f"  Retrieved {len(hits)}/{total} cases")

    rows = []
    for h in hits:
        demo = h.get("demographic", {})
        diag = (h.get("diagnoses") or [{}])[0]
        rows.append({
            "case_id": h.get("case_id"),
            "submitter_id": h.get("submitter_id"),
            "vital_status": demo.get("vital_status"),
            "days_to_death": demo.get("days_to_death"),
            "days_to_last_followup": diag.get("days_to_last_follow_up"),
            "tumor_stage": diag.get("ajcc_pathologic_stage", diag.get("tumor_stage")),
            "prior_treatment": diag.get("prior_treatment"),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "clinical.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} patients → {path}")
    return df


# ---------------------------------------------------------------------------
# 2. Somatic mutations — driver genes
# ---------------------------------------------------------------------------
def download_mutations():
    print("[2] Downloading TCGA-SKCM driver mutations...")
    all_rows = []

    for gene in DRIVER_GENES:
        payload = {
            "filters": {
                "op": "and",
                "content": [
                    {"op": "in", "content": {"field": "case.project.project_id", "value": ["TCGA-SKCM"]}},
                    {"op": "in", "content": {"field": "ssm.consequence.transcript.gene.symbol", "value": [gene]}},
                    {"op": "in", "content": {"field": "ssm.consequence.transcript.annotation.vep_impact",
                                             "value": ["HIGH", "MODERATE"]}},
                ]
            },
            "fields": ",".join([
                "case.submitter_id",
                "ssm.ssm_id",
                "ssm.consequence.transcript.gene.symbol",
                "ssm.consequence.transcript.aa_change",
                "ssm.mutation_type",
                "ssm.genomic_dna_change",
                "ssm.consequence.transcript.annotation.vep_impact",
            ]),
            "format": "JSON",
            "size": "1000",
        }
        try:
            data = _gdc_post("ssm_occurrences", payload, timeout=60)
            hits = data["data"]["hits"]
            count = data["data"]["pagination"]["total"]
            for h in hits:
                csq = (h.get("ssm", {}).get("consequence") or [{}])[0].get("transcript", {})
                all_rows.append({
                    "case_id": h.get("case", {}).get("submitter_id"),
                    "gene": csq.get("gene", {}).get("symbol", gene),
                    "aa_change": csq.get("aa_change", ""),
                    "mutation_type": h.get("ssm", {}).get("mutation_type", ""),
                    "genomic_change": h.get("ssm", {}).get("genomic_dna_change", ""),
                    "vep_impact": csq.get("annotation", {}).get("vep_impact", ""),
                })
            print(f"  {gene:10s}: {count} occurrences")
        except Exception as e:
            print(f"  {gene:10s}: ERROR — {e}")
        time.sleep(0.3)  # polite API usage

    df = pd.DataFrame(all_rows)
    path = os.path.join(OUT_DIR, "driver_mutations.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} mutation records → {path}")
    return df


# ---------------------------------------------------------------------------
# 3. RNA-seq file manifest
# ---------------------------------------------------------------------------
def download_rnaseq_manifest(clinical_df):
    print("[3] Querying RNA-seq file manifest (STAR counts)...")
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-SKCM"]}},
                {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
                {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
                {"op": "in", "content": {"field": "access", "value": ["open"]}},
            ]
        },
        "fields": "file_id,file_name,file_size,cases.submitter_id,cases.case_id",
        "format": "JSON",
        "size": "500",
    }
    data = _gdc_post("files", payload)
    hits = data["data"]["hits"]
    total = data["data"]["pagination"]["total"]

    rows = []
    for h in hits:
        case = (h.get("cases") or [{}])[0]
        rows.append({
            "file_id": h["file_id"],
            "file_name": h["file_name"],
            "file_size_mb": round(h.get("file_size", 0) / 1e6, 1),
            "case_submitter_id": case.get("submitter_id", ""),
            "case_id": case.get("case_id", ""),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "rnaseq_manifest.csv")
    df.to_csv(path, index=False)
    print(f"  Saved manifest: {len(df)}/{total} files → {path}")
    return df


# ---------------------------------------------------------------------------
# 4. Download a small representative RNA-seq subset
#    (10 BRAF V600E + 10 NRAS + 10 NF1 + 10 WT = up to 40 samples)
# ---------------------------------------------------------------------------
def download_rnaseq_subset(mut_df, manifest_df, n_per_subtype: int = 10):
    print("[4] Downloading representative RNA-seq subset...")

    # Build subtype lookup from mutations
    braf_cases = set(mut_df[(mut_df["gene"] == "BRAF") &
                             (mut_df["aa_change"].str.startswith("V600", na=False))]["case_id"])
    nras_cases = set(mut_df[mut_df["gene"] == "NRAS"]["case_id"])
    nf1_cases  = set(mut_df[mut_df["gene"] == "NF1"]["case_id"])
    # WT = none of the above driver mutations
    all_mutated = braf_cases | nras_cases | nf1_cases

    def subtype(cid):
        if cid in braf_cases: return "BRAF_V600E"
        if cid in nras_cases: return "NRAS_mut"
        if cid in nf1_cases:  return "NF1_mut"
        return "Triple_WT"

    manifest_df = manifest_df.copy()
    manifest_df["subtype"] = manifest_df["case_submitter_id"].map(subtype).fillna("Triple_WT")

    selected = []
    for st in ["BRAF_V600E", "NRAS_mut", "NF1_mut", "Triple_WT"]:
        sub = manifest_df[manifest_df["subtype"] == st].head(n_per_subtype)
        selected.append(sub)
        print(f"  {st:12s}: {len(sub)} samples selected")
    sel = pd.concat(selected).reset_index(drop=True)

    # Save download manifest
    sel_path = os.path.join(OUT_DIR, "rnaseq_subset_manifest.csv")
    sel.to_csv(sel_path, index=False)
    print(f"  Subset manifest saved ({len(sel)} files) → {sel_path}")

    # Actually download the files
    rnaseq_dir = os.path.join(OUT_DIR, "rnaseq_subset")
    os.makedirs(rnaseq_dir, exist_ok=True)
    downloaded = 0
    for _, row in sel.iterrows():
        fid = row["file_id"]
        fname = row["file_name"]
        out_path = os.path.join(rnaseq_dir, fname)
        if os.path.exists(out_path):
            downloaded += 1
            continue
        url = f"https://api.gdc.cancer.gov/data/{fid}"
        r = subprocess.run(["curl", "-s", "-o", out_path, url],
                           capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.getsize(out_path) > 1000:
            downloaded += 1
        else:
            if os.path.exists(out_path):
                os.remove(out_path)
        if downloaded % 5 == 0 and downloaded > 0:
            print(f"  Downloaded {downloaded}/{len(sel)} files...")
        time.sleep(0.2)

    print(f"  Downloaded {downloaded}/{len(sel)} RNA-seq files → {rnaseq_dir}/")
    return sel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("=" * 60)
    print("TCGA-SKCM download via GDC API")
    print("=" * 60)

    clinical = download_clinical()
    mutations = download_mutations()
    manifest  = download_rnaseq_manifest(clinical)
    download_rnaseq_subset(mutations, manifest, n_per_subtype=10)

    # Summary
    braf_n = (mutations["gene"] == "BRAF").sum()
    nras_n = (mutations["gene"] == "NRAS").sum()
    nf1_n  = (mutations["gene"] == "NF1").sum()
    print(f"\nMutation summary: BRAF={braf_n} | NRAS={nras_n} | NF1={nf1_n}")
    print(f"Clinical: {len(clinical)} patients")
    print("\nDone. Outputs saved to datas/TCGA-SKCM/")


def main():
    run()


if __name__ == "__main__":
    main()
