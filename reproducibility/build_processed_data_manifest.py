#!/usr/bin/env python3
"""Build a public, label-summary-only manifest for the three historical test sets.

This script records denominators and provenance without exporting residue labels,
model probabilities, embeddings, or third-party model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch


DATASETS = {
    "DNA-129_Test": ("dna", "DNA-129_Test.pt"),
    "DNA-181_Test": ("dna", "DNA-181_Test.pt"),
    "RNA-117_Test": ("rna", "RNA-117_Test.pt"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    obj = torch.load(path, map_location="cpu")
    return obj["rows"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_memory = {}
    for domain in ("dna", "rna"):
        path = args.main_root / f"predictions/memory/full/{domain}.pt"
        for row in load_rows(path):
            base_memory[row["candidate_id"]] = int(row["filtered_hits"])

    extension_memory = {
        row["candidate_id"]: int(row["filtered_hits"])
        for row in load_rows(args.extension_root / "predictions/memory/full/dna.pt")
    }

    base_predictions = set()
    for domain in ("dna", "rna"):
        for row in load_rows(args.main_root / f"predictions/c0/full/{domain}.pt"):
            base_predictions.add(row["candidate_id"])

    complete181_predictions = {
        row["candidate_id"]
        for row in load_rows(args.extension_root / "predictions/combined181/c0_dna181.pt")
    }

    output_rows = []
    for dataset, (domain, filename) in DATASETS.items():
        label_obj = torch.load(args.labels_root / filename, map_location="cpu")
        for label_row in label_obj["rows"]:
            protein_id = label_row["protein_id"]
            y = torch.as_tensor(label_row["y"]).reshape(-1)
            length = int(y.numel())
            positive = int((y == 1).sum().item())
            negative = int((y == 0).sum().item())
            is_extension = dataset == "DNA-181_Test" and protein_id not in base_predictions

            if is_extension:
                structure_path = args.extension_root / f"data/query_pdb/dna/{protein_id}.pdb"
                hits = extension_memory[protein_id]
                protocol = "full_length_long_sequence_extension"
                prediction_present = protein_id in complete181_predictions
            else:
                structure_path = args.main_root / f"data/query_pdb/{domain}/{protein_id}.pdb"
                hits = base_memory[protein_id]
                protocol = "frozen_0916_full_length"
                prediction_present = protein_id in base_predictions

            if not structure_path.exists():
                raise FileNotFoundError(structure_path)
            if not prediction_present:
                raise RuntimeError(f"Missing frozen prediction for {dataset}/{protein_id}")

            output_rows.append(
                {
                    "protein_id": protein_id,
                    "dataset": dataset,
                    "domain": domain.upper(),
                    "sequence_length": length,
                    "label_count": length,
                    "positive_count": positive,
                    "negative_count": negative,
                    "two_class_labels": positive > 0 and negative > 0,
                    "structure_source": "ColabFold/AlphaFold2 predicted monomer rank_001",
                    "structure_sha256": sha256(structure_path),
                    "remote_hit_count": hits,
                    "fallback_status": "exact_SSCP_fallback" if hits == 0 else "remote_evidence_available",
                    "split_or_fold": "historical_external_test",
                    "inference_protocol": protocol,
                    "included_in_primary_evaluation": True,
                }
            )

    output_rows.sort(key=lambda row: (row["dataset"], row["protein_id"]))
    csv_path = args.output_dir / "historical_test_processed_data_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    summaries = {}
    for dataset in DATASETS:
        rows = [row for row in output_rows if row["dataset"] == dataset]
        summaries[dataset] = {
            "proteins": len(rows),
            "residues": sum(row["label_count"] for row in rows),
            "positive_residues": sum(row["positive_count"] for row in rows),
            "two_class_proteins": sum(bool(row["two_class_labels"]) for row in rows),
            "queries_with_remote_evidence": sum(row["remote_hit_count"] > 0 for row in rows),
            "queries_with_exact_fallback": sum(row["remote_hit_count"] == 0 for row in rows),
        }

    summary = {
        "schema": "rimbind_public_processed_data_manifest_v1",
        "scope": "historical external predicted-monomer evaluation",
        "contains_residue_labels": False,
        "contains_model_predictions": False,
        "datasets": summaries,
        "manifest_file": csv_path.name,
        "manifest_sha256": sha256(csv_path),
    }
    summary_path = args.output_dir / "historical_test_processed_data_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    sums_path = args.output_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{sha256(csv_path)}  {csv_path.name}\n"
        f"{sha256(summary_path)}  {summary_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
