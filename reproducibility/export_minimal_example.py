#!/usr/bin/env python3
"""Export a label-free minimal example from frozen RIMBind predictions.

The exporter is intended for release preparation. It never reads benchmark
labels and writes only sequence, predicted-monomer coordinates and model
probabilities for one named query.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_row(path: Path, protein_id: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    return next(row for row in obj["rows"] if row["candidate_id"] == protein_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protein-id", required=True)
    parser.add_argument("--domain", choices=["dna", "rna"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    item_path = args.run_root / f"data/model_inputs/{args.domain}/{args.protein_id}.pt"
    item = torch.load(item_path, map_location="cpu")
    if item.get("contains_labels") is not False:
        raise RuntimeError("example export requires a label-free model input")

    c0 = get_row(args.run_root / f"predictions/c0/full/{args.domain}.pt", args.protein_id)
    final = get_row(args.run_root / f"predictions/final/full/{args.domain}.pt", args.protein_id)
    memory = get_row(args.run_root / f"predictions/memory/full/{args.domain}.pt", args.protein_id)
    saprot = torch.load(args.run_root / f"data/embeddings/saprot/{args.domain}/{args.protein_id}.pt", map_location="cpu")
    esm3 = torch.load(args.run_root / f"data/embeddings/esm3/{args.domain}/{args.protein_id}.pt", map_location="cpu")
    sscp = torch.as_tensor(c0["probability"]).float()
    method = "BAYES_RIM" if args.domain == "dna" else "NO_TEMPLATE_COUNT"
    rimbind = torch.as_tensor(final["probability"][method]).float()
    sequence = item["observed_sequence"]
    if not (len(sequence) == len(sscp) == len(rimbind)):
        raise RuntimeError("sequence and prediction lengths disagree")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = args.output_dir / "example.fasta"
    fasta_path.write_text(f">{args.protein_id}\n{sequence}\n", encoding="utf-8")

    pdb_path = args.output_dir / "example.pdb"
    shutil.copyfile(Path(item["source_pdb"]), pdb_path)
    if sha256(pdb_path) != item["source_pdb_sha256"]:
        raise RuntimeError("copied PDB checksum does not match the frozen input")

    output_path = args.output_dir / "expected_output.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residue_index", "amino_acid", "sscp_probability", "rimbind_probability", "remote_update"])
        for index, (aa, p0, p1) in enumerate(zip(sequence, sscp.tolist(), rimbind.tolist()), 1):
            writer.writerow([index, aa, f"{p0:.9f}", f"{p1:.9f}", f"{p1 - p0:.9f}"])

    fixture_path = args.output_dir / "core_inference_fixture.pt"
    torch.save(
        {
            "schema": "rimbind_label_free_core_smoke_fixture_v1",
            "candidate_id": args.protein_id,
            "domain": args.domain,
            "sequence": sequence,
            "raw_ca_angstrom": item["raw_ca_angstrom"].float().contiguous(),
            "saprot_embedding": saprot["x"].float().contiguous(),
            "esm3_embedding": esm3["x"].float().contiguous(),
            "memory_score": memory["memory_score"].float().contiguous(),
            "template_count": memory["template_count"].int().contiguous(),
            "best_tm": memory["best_tm"].float().contiguous(),
            "contains_labels": False,
        },
        fixture_path,
    )

    sums_path = args.output_dir / "SHA256SUMS"
    files = [fasta_path, pdb_path, output_path, fixture_path]
    sums_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    print(f"exported {args.protein_id}: {len(sequence)} residues")
    print(f"expected_output_sha256={sha256(output_path)}")


if __name__ == "__main__":
    main()
