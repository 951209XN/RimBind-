#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain", choices=["dna", "rna"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    assets = json.loads((root / "audit/MODEL_ASSETS_FROZEN.json").read_text(encoding="utf-8"))
    model_code = Path(assets["files"]["c0_model_code"]["path"])
    if sha256(model_code) != assets["files"]["c0_model_code"]["sha256"]:
        raise RuntimeError("C0 model code changed after freeze")
    sys.path.insert(0, str(model_code.parent))
    from model_single import SingleNABindModel

    manifest = [json.loads(line) for line in (root / "data/manifests/LABEL_FREE_MODEL_INPUTS.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in manifest if row["domain"] == args.domain and row["technical_supported"]]
    if args.scope == "smoke":
        ids = set(json.loads((root / "configs/SMOKE_IDS.json").read_text(encoding="utf-8"))[args.domain])
        rows = [row for row in rows if row["candidate_id"] in ids]
    rows.sort(key=lambda row: row["candidate_id"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("real CUDA is required for C0 prediction")
    checkpoint_rows = {(row["domain"], int(row["fold"])): row for row in assets["c0_checkpoints"]}
    fold_outputs = []

    for fold in range(5):
        checkpoint = Path(checkpoint_rows[(args.domain, fold)]["path"])
        if sha256(checkpoint) != checkpoint_rows[(args.domain, fold)]["sha256"]:
            raise RuntimeError(f"C0 checkpoint changed: {args.domain}/fold{fold}")
        model = SingleNABindModel("C").to(device).eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("variant") != "C":
            raise RuntimeError("checkpoint variant is not C")
        model.load_state_dict(state["state_dict"])
        predictions = []
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                input_path = Path(row["input_path"])
                if sha256(input_path) != row["input_sha256"]:
                    raise RuntimeError(f"input changed: {row['candidate_id']}")
                item = torch.load(input_path, map_location="cpu", weights_only=False)
                if item.get("contains_labels") is not False:
                    raise RuntimeError("C0 input label boundary failure")
                saprot_path = root / "data/embeddings/saprot" / args.domain / f"{row['candidate_id']}.pt"
                esm3_path = root / "data/embeddings/esm3" / args.domain / f"{row['candidate_id']}.pt"
                saprot = torch.load(saprot_path, map_location="cpu", weights_only=False)
                esm3 = torch.load(esm3_path, map_location="cpu", weights_only=False)
                if saprot.get("contains_labels") is not False or esm3.get("contains_labels") is not False:
                    raise RuntimeError("embedding label boundary failure")
                n = int(row["observed_residues"])
                if tuple(saprot["x"].shape) != (n, 1280) or tuple(esm3["x"].shape) != (n, 1536):
                    raise RuntimeError(f"embedding shape mismatch: {row['candidate_id']}")
                logits = model(
                    [saprot["x"].float().to(device)],
                    [item["raw_ca_angstrom"].float().to(device)],
                    [esm3["x"].float().to(device)],
                )
                probability = torch.sigmoid(logits).float().cpu().contiguous()
                if tuple(probability.shape) != (n,) or not torch.isfinite(probability).all() or not bool(((probability >= 0) & (probability <= 1)).all()):
                    raise RuntimeError(f"invalid C0 output: {row['candidate_id']}")
                predictions.append({"candidate_id": row["candidate_id"], "probability": probability, "residues": n})
                print(json.dumps({"scope": args.scope, "domain": args.domain, "fold": fold, "done": index, "total": len(rows), "candidate_id": row["candidate_id"], "device": str(device)}), flush=True)
        out = root / "predictions/c0_folds" / args.scope / args.domain / f"fold{fold}.pt"
        save(out, {
            "schema": "0824_bayesrim_label_free_c0_fold_predictions_v1", "scope": args.scope,
            "domain": args.domain, "fold": fold, "checkpoint_sha256": sha256(checkpoint),
            "model_code_sha256": sha256(model_code), "rows": predictions, "contains_labels": False,
            "metrics_computed": False,
        })
        fold_outputs.append({"fold": fold, "path": str(out), "sha256": sha256(out), "rows": len(predictions)})
        del model
        torch.cuda.empty_cache()

    payloads = [torch.load(row["path"], map_location="cpu", weights_only=False) for row in fold_outputs]
    expected_ids = [row["candidate_id"] for row in payloads[0]["rows"]]
    if any([row["candidate_id"] for row in payload["rows"]] != expected_ids for payload in payloads[1:]):
        raise RuntimeError("C0 fold output order mismatch")
    ensemble = []
    for index, pid in enumerate(expected_ids):
        values = [payload["rows"][index]["probability"] for payload in payloads]
        ensemble.append({"candidate_id": pid, "probability": torch.stack(values).mean(0).contiguous(), "residues": len(values[0])})
    output = root / "predictions/c0" / args.scope / f"{args.domain}.pt"
    save(output, {
        "schema": "0824_bayesrim_label_free_c0_ensemble_v1", "scope": args.scope,
        "domain": args.domain, "ensemble": "arithmetic mean of five strict C0 fold sigmoid probabilities",
        "fold_outputs": fold_outputs, "rows": ensemble, "contains_labels": False, "metrics_computed": False,
    })
    audit = {"complete": True, "scope": args.scope, "domain": args.domain, "proteins": len(ensemble), "output": str(output), "sha256": sha256(output), "device": str(device), "labels_read": False}
    audit_path = root / "audit" / f"C0_{args.scope.upper()}_{args.domain}.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
