#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


HISTORICAL_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def crop(hidden: torch.Tensor, n: int) -> torch.Tensor:
    hidden = hidden.squeeze(0).detach().float().cpu()
    if hidden.size(0) == n + 2:
        return hidden[1:-1]
    if hidden.size(0) == n:
        return hidden
    raise RuntimeError(f"embedding length {hidden.size(0)} != {n} or {n}+2")


def historical_item_sequence(item: dict) -> str:
    sequence = []
    for key in item.get("residue_keys", []):
        residue_name = str(key[1]).upper() if isinstance(key, (tuple, list)) and len(key) >= 2 else ""
        sequence.append(HISTORICAL_AA3_TO_1.get(residue_name, "X"))
    return "".join(sequence)


def selected_rows(root: Path, scope: str, domain: str | None) -> list[dict]:
    rows = [json.loads(line) for line in (root / "data/manifests/LABEL_FREE_MODEL_INPUTS.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["technical_supported"] and (domain is None or row["domain"] == domain)]
    if scope == "smoke":
        smoke = json.loads((root / "configs/SMOKE_IDS.json").read_text(encoding="utf-8"))
        ids = set(smoke[domain]) if domain else set(smoke["dna"] + smoke["rna"])
        rows = [row for row in rows if row["candidate_id"] in ids]
    return sorted(rows, key=lambda row: (row["domain"], row["candidate_id"]))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--kind", choices=["saprot", "esm3"], required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--scope", choices=["smoke", "full"], required=True)
    parser.add_argument("--domain", choices=["dna", "rna"])
    parser.add_argument("--saprot-model", help="Local SaProt_650M_AF2 directory; required for --kind saprot")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    assets = json.loads((root / "audit/MODEL_ASSETS_FROZEN.json").read_text(encoding="utf-8"))
    rows = selected_rows(root, args.scope, args.domain)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("real CUDA is required for embedding generation")

    if args.kind == "saprot":
        from transformers import EsmModel, EsmTokenizer
        model_setting = args.saprot_model or os.environ.get("SAPROT_MODEL_DIR")
        if not model_setting:
            raise RuntimeError("provide --saprot-model or set SAPROT_MODEL_DIR")
        model_path = Path(model_setting).expanduser().resolve()
        tokenizer = EsmTokenizer.from_pretrained(model_path, local_files_only=True)
        model = EsmModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    else:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        model = ESM3.from_pretrained("esm3_sm_open_v1", device=device).eval()

    outputs = []
    for index, row in enumerate(rows, 1):
        input_path = Path(row["input_path"])
        if sha256(input_path) != row["input_sha256"]:
            raise RuntimeError(f"model input changed: {row['candidate_id']}")
        item = torch.load(input_path, map_location="cpu", weights_only=False)
        if item.get("contains_labels") is not False or item.get("nucleic_acid_coordinates_read") is not False:
            raise RuntimeError("label boundary failure")
        pid, sequence, n = row["candidate_id"], item["observed_sequence"], item["observed_residues"]
        out = root / "data/embeddings" / args.kind / row["domain"] / f"{pid}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            cached = torch.load(out, map_location="cpu", weights_only=False)
            if cached.get("candidate_id") == pid and tuple(cached.get("x", torch.empty(0)).shape) == (n, 1280 if args.kind == "saprot" else 1536) and cached.get("contains_labels") is False:
                outputs.append({"candidate_id": pid, "path": str(out), "sha256": sha256(out), "cached": True})
                continue
        if args.kind == "saprot":
            three_di = (root / "data/embeddings/3di" / row["domain"] / f"{pid}.3di").read_text(encoding="utf-8").strip()
            if len(three_di) != n:
                raise RuntimeError(f"3Di mismatch: {pid}")
            text = "".join(a + b.lower() for a, b in zip(sequence, three_di))
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=False)
            if int(tokens["input_ids"].shape[1]) != n + 2:
                raise RuntimeError(f"SaProt token length mismatch: {pid}")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            x = crop(model(**tokens).last_hidden_state, n)
        else:
            source_mode = "pdb_sequence_structure"
            fallback_reason = None
            try:
                protein = ESMProtein.from_pdb(Path(item["source_pdb"]), chain_id="A")
                if len(protein) != n:
                    fallback_reason = f"pdb_length_{len(protein)}_item_length_{n}"
                    protein = ESMProtein(sequence=historical_item_sequence(item))
                    source_mode = "item_sequence_length_mismatch"
            except Exception as exc:
                fallback_reason = f"pdb_failed_{type(exc).__name__}"
                protein = ESMProtein(sequence=historical_item_sequence(item))
                source_mode = "item_sequence_pdb_failed"
            if len(protein) != n:
                raise RuntimeError(f"ESM3 historical fallback length mismatch: {pid}")
            encoded = model.encode(protein).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    sequence_tokens=encoded.sequence.unsqueeze(0) if encoded.sequence is not None else None,
                    structure_tokens=encoded.structure.unsqueeze(0) if encoded.structure is not None else None,
                    ss8_tokens=encoded.secondary_structure.unsqueeze(0) if encoded.secondary_structure is not None else None,
                    sasa_tokens=encoded.sasa.unsqueeze(0) if encoded.sasa is not None else None,
                    function_tokens=encoded.function.unsqueeze(0) if encoded.function is not None else None,
                    residue_annotation_tokens=encoded.residue_annotations.unsqueeze(0) if encoded.residue_annotations is not None else None,
                    structure_coords=encoded.coordinates.unsqueeze(0) if encoded.coordinates is not None else None,
                )
                x = crop(output.embeddings, n)
        if tuple(x.shape) != (n, 1280 if args.kind == "saprot" else 1536) or not torch.isfinite(x).all():
            raise RuntimeError(f"invalid {args.kind} embedding: {pid}/{tuple(x.shape)}")
        payload = {
            "schema": "0824_bayesrim_label_free_embedding_v1", "kind": args.kind,
            "candidate_id": pid, "domain": row["domain"], "x": x.half().contiguous(),
            "input_sha256": row["input_sha256"], "contains_labels": False,
            "nucleic_acid_coordinates_read": False,
            "protein_only_source_pdb_sha256": sha256(Path(item["source_pdb"])),
            "source_mode": "full_precision_saprot" if args.kind == "saprot" else source_mode,
            "fallback_reason": None if args.kind == "saprot" else fallback_reason,
        }
        temporary = out.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, out)
        outputs.append({"candidate_id": pid, "path": str(out), "sha256": sha256(out), "cached": False})
        print(json.dumps({"kind": args.kind, "scope": args.scope, "done": index, "total": len(rows), "candidate_id": pid, "residues": n, "device": str(device)}), flush=True)

    audit = {
        "schema": "0824_bayesrim_label_free_embeddings_audit_v1", "kind": args.kind,
        "scope": args.scope, "domain": args.domain, "device": str(device), "rows": outputs,
        "labels_read": False, "nucleic_acid_coordinates_read": False, "complete": len(outputs) == len(rows),
    }
    suffix = f"_{args.domain}" if args.domain else ""
    path = root / "audit" / f"EMBEDDINGS_{args.kind.upper()}_{args.scope.upper()}{suffix}.json"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": audit["complete"], "kind": args.kind, "rows": len(outputs), "audit": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
