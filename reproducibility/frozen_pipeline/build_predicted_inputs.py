#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "MSE":"M","SEC":"U","PYL":"O","ASX":"B","GLX":"Z",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse(path: Path) -> tuple[list[tuple[str, str, str, str]], str, torch.Tensor]:
    keys, sequence, coords, seen = [], [], [], set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54 or line[12:16].strip() != "CA":
            continue
        residue = line[17:20].strip().upper()
        if residue not in AA3_TO_1:
            continue
        key = (line[21].strip() or "A", residue, line[22:26].strip(), line[26].strip())
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
        sequence.append(AA3_TO_1[residue])
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    ca = torch.tensor(coords, dtype=torch.float32)
    if tuple(ca.shape) != (len(keys), 3) or not torch.isfinite(ca).all():
        raise RuntimeError(f"invalid predicted coordinates: {path}")
    return keys, "".join(sequence), ca


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = root / "data/manifests" / f"PREDICTED_MONOMER_{args.scope.upper()}.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_rows = []
    smoke = {"dna": [], "rna": []}
    for row in rows:
        pid, domain = row["protein_id"], row["domain"]
        pdb = Path(row["query_pdb"])
        if sha256(pdb) != row["query_pdb_sha256"]:
            raise RuntimeError(f"PDB changed: {pid}")
        keys, sequence, ca = parse(pdb)
        if sequence != row["sequence"] or len(sequence) > 1022:
            raise RuntimeError(f"sequence or length mismatch: {pid}")
        output = root / "data/model_inputs" / domain / f"{pid}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "0916_rimbind_predicted_monomer_input_v1",
            "candidate_id": pid,
            "dataset": row["dataset"],
            "domain": domain,
            "residue_keys": keys,
            "observed_sequence": sequence,
            "raw_ca_angstrom": ca,
            "observed_residues": len(sequence),
            "source_pdb": str(pdb),
            "source_pdb_sha256": row["query_pdb_sha256"],
            "structure_source": "ColabFold AlphaFold2 ptm predicted monomer without templates",
            "mean_plddt": row["mean_plddt"],
            "contains_labels": False,
            "nucleic_acid_coordinates_read": False,
        }
        temporary = output.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        output_rows.append({
            "candidate_id": pid,
            "dataset": row["dataset"],
            "domain": domain,
            "observed_residues": len(sequence),
            "technical_supported": True,
            "input_path": str(output),
            "input_sha256": sha256(output),
            "contains_labels": False,
        })
        smoke[domain].append(pid)
    manifest = root / "data/manifests/LABEL_FREE_MODEL_INPUTS.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(output_rows, key=lambda x: (x["domain"], x["candidate_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (root / "configs/SMOKE_IDS.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema": "0916_rimbind_predicted_monomer_inputs_v1",
        "scope": args.scope,
        "proteins": len(output_rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "labels_read": False,
        "complete": True,
    }
    path = root / "audit" / f"PREDICTED_INPUTS_{args.scope.upper()}.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

