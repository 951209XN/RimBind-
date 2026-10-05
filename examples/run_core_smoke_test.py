#!/usr/bin/env python3
"""Run the frozen five-fold SSCP + BayesRIM core on a prepared fixture.

This validates the model and fusion stages without downloading or rerunning
SaProt, ESM3, ColabFold or Foldseek. It is not a replacement for the complete
sequence/PDB preprocessing workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_evidence(memory: np.ndarray, count: np.ndarray, tm: np.ndarray) -> np.ndarray:
    prior, tau, exponent = 0.10, 1.0, 2.0
    covered = count > 0
    posterior = np.clip((count * memory + tau * prior) / (count + tau), 1e-4, 1 - 1e-4)
    value = np.log(posterior / (1 - posterior)) - math.log(prior / (1 - prior))
    value[~covered] = 0.0
    return value * np.power(tm, exponent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.model_code.parent.resolve()))
    from model_single import SingleNABindModel

    fixture = torch.load(args.fixture, map_location="cpu")
    if fixture.get("contains_labels") is not False:
        raise RuntimeError("fixture must be label-free")
    domain = fixture["domain"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fold_probabilities = []
    for fold in range(5):
        checkpoint = args.checkpoint_root / f"checkpoints/{domain}/fold{fold}/best.pt"
        state = torch.load(checkpoint, map_location="cpu")
        if state.get("variant") != "C":
            raise RuntimeError(f"unexpected checkpoint variant in fold {fold}")
        model = SingleNABindModel("C").to(device).eval()
        model.load_state_dict(state["state_dict"])
        with torch.inference_mode():
            logits = model(
                [fixture["saprot_embedding"].to(device)],
                [fixture["raw_ca_angstrom"].to(device)],
                [fixture["esm3_embedding"].to(device)],
            )
        fold_probabilities.append(torch.sigmoid(logits).float().cpu())

    p_base = torch.stack(fold_probabilities).mean(0).numpy().astype(np.float64)
    p_clip = np.clip(p_base, 1e-6, 1 - 1e-6)
    z_base = np.log(p_clip / (1 - p_clip))
    memory = np.nan_to_num(fixture["memory_score"].numpy().astype(np.float64), nan=0.0)
    count = fixture["template_count"].numpy().astype(np.float64)
    tm = fixture["best_tm"].numpy().astype(np.float64)
    p_final = (1 / (1 + np.exp(-np.clip(z_base + 0.2 * remote_evidence(memory, count, tm), -30, 30)))).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["residue_index", "amino_acid", "sscp_probability", "rimbind_probability", "remote_update"])
        for index, (aa, p0, p1) in enumerate(zip(fixture["sequence"], p_base, p_final), 1):
            writer.writerow([index, aa, f"{p0:.9f}", f"{p1:.9f}", f"{p1 - p0:.9f}"])

    print(f"output={args.output}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
