#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


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


def pdb_sequence(path: Path) -> tuple[str, float]:
    seen = set()
    sequence = []
    plddt = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66 or line[12:16].strip() != "CA":
            continue
        key = (line[21].strip(), line[22:26].strip(), line[26].strip())
        if key in seen:
            continue
        seen.add(key)
        residue = line[17:20].strip().upper()
        if residue not in AA3_TO_1:
            raise RuntimeError(f"unknown residue {residue} in {path}")
        sequence.append(AA3_TO_1[residue])
        plddt.append(float(line[60:66]))
    if not sequence:
        raise RuntimeError(f"no CA atoms in {path}")
    return "".join(sequence), sum(plddt) / len(plddt)


def choose_pdb(output: Path, pid: str) -> Path:
    patterns = [
        f"{pid}_relaxed_rank_001_*.pdb",
        f"{pid}_unrelaxed_rank_001_*.pdb",
        f"{pid}_rank_001_*.pdb",
    ]
    for pattern in patterns:
        matches = sorted(output.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"no rank-001 PDB for {pid} in {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest_path = root / "data/sequences" / args.scope / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = root / "runs/colabfold" / args.scope
    collected = []
    for row in rows:
        pid = row["protein_id"]
        pdb = choose_pdb(source, pid)
        observed, mean_plddt = pdb_sequence(pdb)
        if observed != row["sequence"]:
            raise RuntimeError(f"predicted sequence mismatch: {pid}, expected {len(row['sequence'])}, observed {len(observed)}")
        target = root / "data/query_pdb" / row["domain"] / f"{pid}.pdb"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdb, target)
        collected.append({
            **row,
            "source_pdb": str(pdb),
            "source_pdb_sha256": sha256(pdb),
            "query_pdb": str(target),
            "query_pdb_sha256": sha256(target),
            "mean_plddt": mean_plddt,
        })
    out = root / "data/manifests" / f"PREDICTED_MONOMER_{args.scope.upper()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sorted(collected, key=lambda x: (x["domain"], x["protein_id"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {
        "schema": "0916_rimbind_colabfold_collection_v1",
        "scope": args.scope,
        "proteins": len(collected),
        "mean_plddt": sum(x["mean_plddt"] for x in collected) / len(collected),
        "manifest": str(out),
        "manifest_sha256": sha256(out),
        "labels_read": False,
        "complete": True,
    }
    audit_path = root / "audit" / f"COLABFOLD_COLLECTION_{args.scope.upper()}.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

