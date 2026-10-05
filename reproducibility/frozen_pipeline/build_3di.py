#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed: {command}; see {log}")


def fasta(path: Path) -> dict[str, str]:
    result, name, body = {}, None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if name is not None:
                result[name] = "".join(body)
            name, body = line[1:].split()[0], []
        elif line:
            body.append(line)
    if name is not None:
        result[name] = "".join(body)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    assets = json.loads((root / "audit/MODEL_ASSETS_FROZEN.json").read_text(encoding="utf-8"))
    foldseek = Path(assets["files"]["foldseek_binary"]["path"])
    if sha256(foldseek) != assets["files"]["foldseek_binary"]["sha256"]:
        raise RuntimeError("Foldseek changed after asset freeze")
    summary = {}
    for domain in ("dna", "rna"):
        source = root / "data/query_pdb" / domain
        work = root / "runs/3di" / domain
        work.mkdir(parents=True, exist_ok=True)
        db = work / "query"
        run([str(foldseek), "createdb", str(source), str(db), "--threads", "16", "-v", "1"], root / "logs" / f"3di_createdb_{domain}.log")
        ss_fasta = work / "three_di.fasta"
        run([str(foldseek), "lndb", str(db) + "_h", str(db) + "_ss_h", "-v", "1"], root / "logs" / f"3di_lndb_{domain}.log")
        run([str(foldseek), "convert2fasta", str(db) + "_ss", str(ss_fasta), "-v", "1"], root / "logs" / f"3di_fasta_{domain}.log")
        values = fasta(ss_fasta)
        inputs = [json.loads(line) for line in (root / "data/manifests/LABEL_FREE_MODEL_INPUTS.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        inputs = [row for row in inputs if row["domain"] == domain]
        out_dir = root / "data/embeddings/3di" / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for row in inputs:
            pid = row["candidate_id"]
            seq = values.get(pid)
            if seq is None:
                missing.append(pid)
                continue
            if len(seq) != int(row["observed_residues"]):
                raise RuntimeError(f"3Di length mismatch: {pid}")
            path = out_dir / f"{pid}.3di"
            temporary = path.with_suffix(".3di.tmp")
            temporary.write_text(seq + "\n", encoding="utf-8")
            os.replace(temporary, path)
        if missing:
            raise RuntimeError(f"{domain}: missing 3Di for {len(missing)} candidates, first={missing[:5]}")
        summary[domain] = {"sequences": len(inputs), "fasta_sha256": sha256(ss_fasta)}
    output = {"schema": "0824_bayesrim_3di_audit_v1", "domains": summary, "labels_read": False, "complete": True}
    path = root / "audit/THREE_DI.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
