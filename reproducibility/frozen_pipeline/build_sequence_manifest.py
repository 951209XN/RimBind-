#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def records(path: Path) -> list[dict]:
    result: list[dict] = []
    name = None
    body: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                # Historical benchmark files store one sequence line followed by
                # one residue-label line.  Structure prediction must consume only
                # the first line; labels are deliberately ignored here.
                result.append({"protein_id": name, "sequence": body[0]})
            name, body = line[1:].split()[0], []
        else:
            body.append(line)
    if name is not None:
        result.append({"protein_id": name, "sequence": body[0]})
    if any(not row["sequence"] or any(x not in "ACDEFGHIKLMNPQRSTVWYBXZUO" for x in row["sequence"]) for row in result):
        raise RuntimeError(f"invalid amino-acid sequence in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cfg = json.loads((root / "configs/mvp_protocol.json").read_text(encoding="utf-8"))
    workspace = Path(cfg["workspace"])
    max_residues = int(cfg["predicted_monomer"]["max_residues"])
    all_rows: list[dict] = []
    excluded_rows: list[dict] = []
    selected_ids: list[str] = []
    for dataset, spec in cfg["datasets"].items():
        source = workspace / spec["raw_manifest"]
        rows = records(source)
        rows = sorted(rows, key=lambda row: row["protein_id"])
        for row in rows:
            row.update({"dataset": dataset, "domain": spec["domain"], "length": len(row["sequence"]), "source_manifest": str(source)})
        excluded_rows.extend({**row, "exclusion_reason": f"length_gt_{max_residues}"} for row in rows if row["length"] > max_residues)
        rows = [row for row in rows if row["length"] <= max_residues]
        if args.scope == "smoke":
            target = 180
            rows = [min(rows, key=lambda row: (abs(len(row["sequence"]) - target), row["protein_id"]))]
        for row in rows:
            all_rows.append(row)
            selected_ids.append(row["protein_id"])

    duplicate_ids = sorted({pid for pid in selected_ids if selected_ids.count(pid) > 1})
    if duplicate_ids:
        raise RuntimeError(f"protein IDs overlap across datasets: {duplicate_ids[:10]}")

    out_dir = root / "data/sequences" / args.scope
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta = out_dir / "queries.fasta"
    with fasta.open("w", encoding="utf-8") as handle:
        for row in sorted(all_rows, key=lambda row: (row["length"], row["protein_id"])):
            handle.write(f">{row['protein_id']}\n{row['sequence']}\n")
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(all_rows, key=lambda row: (row["dataset"], row["protein_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    exclusions = out_dir / "technical_exclusions.jsonl"
    with exclusions.open("w", encoding="utf-8") as handle:
        for row in sorted(excluded_rows, key=lambda row: (row["dataset"], row["protein_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema": "0916_rimbind_predicted_monomer_sequence_manifest_v1",
        "scope": args.scope,
        "proteins": len(all_rows),
        "residues": sum(row["length"] for row in all_rows),
        "max_residues": max_residues,
        "technical_exclusions": len(excluded_rows),
        "technical_exclusions_by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in excluded_rows)
            for dataset in cfg["datasets"]
        },
        "exclusions": str(exclusions),
        "exclusions_sha256": sha256(exclusions),
        "ids": selected_ids,
        "fasta": str(fasta),
        "fasta_sha256": sha256(fasta),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "labels_used": False,
        "source_benchmark_files_contain_labels": True,
        "complete": True,
    }
    audit_path = root / "audit" / f"SEQUENCE_MANIFEST_{args.scope.upper()}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
