#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


COLUMNS = ("query","target","fident","alnlen","qstart","qend","tstart","tend","evalue","bits","qcov","tcov","alntmscore","qtmscore","ttmscore","qaln","taln")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def logged(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed; see {log}")


def pdb_accession(pid: str) -> str:
    return pid.split("_", 1)[0].lower()


def parse_alignments(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) != len(COLUMNS):
                raise RuntimeError(f"bad Foldseek row width: {len(fields)}")
            row = dict(zip(COLUMNS, fields))
            for key in ("fident","evalue","bits","qcov","tcov","alntmscore","qtmscore","ttmscore"):
                row[key] = float(row[key])
            for key in ("alnlen","qstart","qend","tstart","tend"):
                row[key] = int(row[key])
            grouped[row["query"]].append(row)
    return grouped


def passes(row: dict) -> bool:
    return (
        pdb_accession(row["query"]) != pdb_accession(row["target"])
        and row["fident"] <= 0.30 and row["alnlen"] >= 30
        and row["qcov"] >= 0.30 and row["tcov"] >= 0.20
        and row["alntmscore"] >= 0.40 and row["evalue"] <= 10.0
    )


def rotate(values: np.ndarray, pid: str) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    shift = 1 + int.from_bytes(hashlib.sha256(pid.encode()).digest()[:8], "little") % (len(values) - 1)
    return np.roll(values, shift)


def project(row: dict, qn: int, labels: np.ndarray, sums: np.ndarray, rotated: np.ndarray, weights: np.ndarray, counts: np.ndarray, best_tm: np.ndarray) -> None:
    qi, ti = row["qstart"] - 1, row["tstart"] - 1
    q_aln, t_aln = row["qaln"], row["taln"]
    if len(q_aln) != len(t_aln):
        raise RuntimeError("alignment string mismatch")
    weight = row["alntmscore"] ** 2 * row["qcov"]
    rotated_labels = rotate(labels, row["target"])
    for qc, tc in zip(q_aln, t_aln):
        qidx = qi if qc != "-" else None
        tidx = ti if tc != "-" else None
        if qidx is not None and tidx is not None:
            if not (0 <= qidx < qn and 0 <= tidx < len(labels)):
                raise RuntimeError(f"alignment index failure: {row['query']}->{row['target']}")
            sums[qidx] += weight * labels[tidx]
            rotated[qidx] += weight * rotated_labels[tidx]
            weights[qidx] += weight
            counts[qidx] += 1
            best_tm[qidx] = max(best_tm[qidx], row["alntmscore"])
        if qc != "-":
            qi += 1
        if tc != "-":
            ti += 1
    if qi != row["qend"] or ti != row["tend"]:
        raise RuntimeError("inclusive endpoint mismatch")


def evidence(memory: np.ndarray, count: np.ndarray, tm: np.ndarray, prior: float, tau: float, exponent: float, mode: str = "FULL") -> np.ndarray:
    covered = count > 0
    effective = covered.astype(np.float64) if mode == "NO_TEMPLATE_COUNT" else count.astype(np.float64)
    posterior = np.clip((effective * memory + tau * prior) / (effective + tau), 1e-4, 1 - 1e-4)
    value = np.log(posterior / (1 - posterior)) - np.log(prior / (1 - prior))
    value[~covered] = 0
    if mode == "POSITIVE_ONLY":
        value = np.maximum(value, 0)
    elif mode == "NEGATIVE_ONLY":
        value = np.minimum(value, 0)
    if mode != "NO_TM_CONFIDENCE":
        value *= np.power(tm, exponent)
    return value


def sigmoid(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((1 / (1 + np.exp(-np.clip(values, -30, 30)))).astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain", choices=["dna", "rna"], required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    parser.add_argument("--memory-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    assets = json.loads((root / "audit/MODEL_ASSETS_FROZEN.json").read_text(encoding="utf-8"))
    foldseek = Path(assets["files"]["foldseek_binary"]["path"])
    if sha256(foldseek) != assets["files"]["foldseek_binary"]["sha256"]:
        raise RuntimeError("Foldseek changed after freeze")
    memory_path = Path(assets["files"][f"training_memory_{args.domain}"]["path"])
    if sha256(memory_path) != assets["files"][f"training_memory_{args.domain}"]["sha256"]:
        raise RuntimeError("training memory changed after freeze")
    memory_payload = torch.load(memory_path, map_location="cpu", weights_only=False)
    training_labels = {row["protein_id"]: row["labels"].numpy().astype(np.uint8) for row in memory_payload["rows"]}

    manifest = [json.loads(line) for line in (root / "data/manifests/LABEL_FREE_MODEL_INPUTS.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in manifest if row["domain"] == args.domain and row["technical_supported"]]
    if args.scope == "smoke":
        smoke = set(json.loads((root / "configs/SMOKE_IDS.json").read_text(encoding="utf-8"))[args.domain])
        rows = [row for row in rows if row["candidate_id"] in smoke]
    rows.sort(key=lambda row: row["candidate_id"])

    query_dir = root / "runs/foldseek/query_pdb" / args.scope / args.domain
    query_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = root / "data/query_pdb" / args.domain / f"{row['candidate_id']}.pdb"
        link = query_dir / source.name
        if not link.exists():
            os.symlink(source, link)

    target_source = Path("/home/ys/xuning/MyProject/0814-BayesRIM-FrozenExternalConfirmation/data/memory_pdb") / args.domain
    db_root = root / "runs/foldseek/db"
    target_db = db_root / "memory" / args.domain
    query_db = db_root / "query" / args.scope / args.domain
    target_db.parent.mkdir(parents=True, exist_ok=True)
    query_db.parent.mkdir(parents=True, exist_ok=True)
    if not Path(str(target_db) + ".dbtype").is_file():
        logged([str(foldseek), "createdb", str(target_source), str(target_db), "--threads", "8", "-v", "1"], root / "logs" / f"v1_createdb_memory_{args.domain}.log")
    if not Path(str(query_db) + ".dbtype").is_file():
        logged([str(foldseek), "createdb", str(query_dir), str(query_db), "--threads", "8", "-v", "1"], root / "logs" / f"v1_createdb_query_{args.scope}_{args.domain}.log")
    task = root / "runs/foldseek/search" / args.scope / args.domain
    task.mkdir(parents=True, exist_ok=True)
    result_db, tmp, alignment_path = task / "result", task / "tmp", task / "alignments.tsv"
    tmp.mkdir(exist_ok=True)
    if not alignment_path.is_file():
        logged([str(foldseek), "search", str(query_db), str(target_db), str(result_db), str(tmp), "-s", "9.5", "--max-seqs", "1000", "--alignment-type", "2", "--exact-tmscore", "1", "-a", "1", "-e", "10", "--threads", "16", "-v", "1"], root / "logs" / f"v1_search_{args.scope}_{args.domain}.log")
        temporary = alignment_path.with_suffix(".tsv.tmp")
        logged([str(foldseek), "convertalis", str(query_db), str(target_db), str(result_db), str(temporary), "--format-output", ",".join(COLUMNS), "--exact-tmscore", "1", "--threads", "8", "-v", "1"], root / "logs" / f"v1_convert_{args.scope}_{args.domain}.log")
        os.replace(temporary, alignment_path)

    grouped = parse_alignments(alignment_path)
    memory_rows = []
    for row in rows:
        item = torch.load(row["input_path"], map_location="cpu", weights_only=False)
        if item.get("contains_labels") is not False:
            raise RuntimeError("query label boundary failure")
        pid, n = row["candidate_id"], int(row["observed_residues"])
        hits = [hit for hit in grouped.get(pid, []) if hit["target"] in training_labels and passes(hit)]
        hits.sort(key=lambda hit: (hit["alntmscore"], hit["qcov"], hit["bits"]), reverse=True)
        hits = hits[:20]
        sums = np.zeros(n); rotated_sums = np.zeros(n); weights = np.zeros(n)
        counts = np.zeros(n, np.int32); best_tm = np.zeros(n, np.float32)
        for hit in hits:
            project(hit, n, training_labels[hit["target"]], sums, rotated_sums, weights, counts, best_tm)
        covered = weights > 0
        score = np.full(n, np.nan, np.float32); control = np.full(n, np.nan, np.float32)
        score[covered] = (sums[covered] / weights[covered]).astype(np.float32)
        control[covered] = (rotated_sums[covered] / weights[covered]).astype(np.float32)
        memory_rows.append({"candidate_id": pid, "memory_score": torch.from_numpy(score), "rotated_score": torch.from_numpy(control), "template_count": torch.from_numpy(counts), "best_tm": torch.from_numpy(best_tm), "filtered_hits": len(hits)})

    memory_output = root / "predictions/memory" / args.scope / f"{args.domain}.pt"
    atomic_torch(memory_output, {"schema": "0824_bayesrim_external_memory_v1", "scope": args.scope, "domain": args.domain, "rows": memory_rows, "alignment_sha256": sha256(alignment_path), "query_labels_read": False, "contains_external_labels": False})

    if args.memory_only:
        audit = {
            "schema": "0824_bayesrim_v1_memory_audit_v1", "scope": args.scope, "domain": args.domain,
            "proteins": len(memory_rows), "queries_with_hit": sum(row["filtered_hits"] > 0 for row in memory_rows),
            "covered_residues": sum(int(torch.isfinite(row["memory_score"]).sum()) for row in memory_rows),
            "total_residues": sum(len(row["memory_score"]) for row in memory_rows),
            "alignment_sha256": sha256(alignment_path), "memory_sha256": sha256(memory_output),
            "labels_read": False, "metrics_computed": False, "complete": True,
        }
        atomic_json(root / "audit" / f"V1_MEMORY_{args.scope.upper()}_{args.domain}.json", audit)
        print(json.dumps(audit, ensure_ascii=False), flush=True)
        return

    c0_path = root / "predictions/c0" / args.scope / f"{args.domain}.pt"
    c0 = torch.load(c0_path, map_location="cpu", weights_only=False)
    if c0.get("contains_labels") is not False:
        raise RuntimeError("C0 label boundary failure")
    c0_by_id = {row["candidate_id"]: row for row in c0["rows"]}
    memory_by_id = {row["candidate_id"]: row for row in memory_rows}
    if set(c0_by_id) != set(memory_by_id):
        raise RuntimeError("C0/memory coverage mismatch")
    parameters = {"dna": {"bayes_alpha": 0.2, "bayes_tau": 1.0, "rim_alpha": 0.25}, "rna": {"bayes_alpha": 0.8, "bayes_tau": 4.0, "rim_alpha": 1.0}}[args.domain]
    final_rows = []
    for pid in sorted(c0_by_id):
        p = np.clip(c0_by_id[pid]["probability"].numpy().astype(np.float64), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p)); memory = memory_by_id[pid]
        m = np.nan_to_num(memory["memory_score"].numpy().astype(np.float64), nan=0.0)
        rotated = np.nan_to_num(memory["rotated_score"].numpy().astype(np.float64), nan=0.0)
        count = memory["template_count"].numpy().astype(np.float64); tm = memory["best_tm"].numpy().astype(np.float64)
        full = evidence(m, count, tm, 0.10, parameters["bayes_tau"], 2.0)
        methods = {"C0": torch.from_numpy(p.astype(np.float32)), "ORIGINAL_RIM": sigmoid(z + parameters["rim_alpha"] * m), "BAYES_RIM": sigmoid(z + parameters["bayes_alpha"] * full)}
        for mode in ("POSITIVE_ONLY", "NEGATIVE_ONLY", "NO_TM_CONFIDENCE", "NO_TEMPLATE_COUNT"):
            methods[mode] = sigmoid(z + parameters["bayes_alpha"] * evidence(m, count, tm, 0.10, parameters["bayes_tau"], 2.0, mode))
        methods["ROTATED_LABELS"] = sigmoid(z + parameters["bayes_alpha"] * evidence(rotated, count, tm, 0.10, parameters["bayes_tau"], 2.0))
        if any(not torch.isfinite(value).all() or not bool(((value >= 0) & (value <= 1)).all()) for value in methods.values()):
            raise RuntimeError(f"invalid V1 probability: {pid}")
        final_rows.append({"candidate_id": pid, "probability": methods})
    final_output = root / "predictions/final" / args.scope / f"{args.domain}.pt"
    atomic_torch(final_output, {"schema": "0824_bayesrim_label_free_final_predictions_v1", "scope": args.scope, "domain": args.domain, "fixed_parameters": parameters, "rows": final_rows, "contains_labels": False, "external_metrics_computed": False})
    audit = {
        "schema": "0824_bayesrim_v1_prediction_audit_v1", "scope": args.scope, "domain": args.domain,
        "proteins": len(final_rows), "queries_with_hit": sum(row["filtered_hits"] > 0 for row in memory_rows),
        "covered_residues": sum(int(torch.isfinite(row["memory_score"]).sum()) for row in memory_rows),
        "total_residues": sum(len(row["memory_score"]) for row in memory_rows),
        "alignment_sha256": sha256(alignment_path), "memory_sha256": sha256(memory_output),
        "final_prediction_sha256": sha256(final_output), "labels_read": False, "metrics_computed": False, "complete": True,
    }
    atomic_json(root / "audit" / f"V1_{args.scope.upper()}_{args.domain}.json", audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
