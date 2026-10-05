# Inference Guide

## Scope

The audited pipeline has four stages: predicted-monomer input assembly, frozen SaProt and ESM3 embedding generation, five-fold SSCP inference, and Foldseek-based remote-memory projection followed by BayesRIM fusion. The files in `reproducibility/frozen_pipeline` are the exact provenance snapshot; they enforce hashes and the original directory contract. They are supplied for traceability, not as a one-command portable installer.

## Required software

- Linux with an NVIDIA GPU for the frozen neural encoders.
- Python and PyTorch compatible with the recorded environments.
- SaProt 650M AF2 and Foldseek for amino-acid plus 3Di embeddings and structural retrieval.
- ESM package version 3.1.1 and the `esm3_sm_open_v1` checkpoint.
- Python packages used by the frozen model, including `torch`, `transformers`, `numpy`, `biopython` and `mamba-ssm`.

The audited server environments were:

| Component | PyTorch | Transformers | CUDA |
|---|---:|---:|---:|
| SSCP and SaProt | 2.1.1 | 4.43.3 | 11.8 |
| ESM3 | 2.10.0+cu128 | 4.36.2 | 12.8 |

Exact third-party model and binary hashes are listed in `checkpoints/manifest.json`.

## Required RIMBind assets

The Zenodo archive must contain:

1. ten task-specific SSCP checkpoints, five for DNA and five for RNA;
2. the DNA and RNA training-memory label files;
3. a portable `MODEL_ASSETS_FROZEN.json` whose relative paths resolve inside the extracted archive;
4. the query sequence manifest and predicted-monomer PDB files, or instructions to regenerate them;
5. an example input and its expected output checksum.

## Query input contract

Each query requires a unique identifier, amino-acid sequence and protein-only PDB coordinate file. Residue order and length must agree across the sequence, coordinate-derived model input, SaProt embedding and ESM3 embedding. Query nucleic-acid coordinates, query contact labels and evaluation labels are not read during prediction.

## Frozen order of computation

1. Build the protein-only input tensors and Foldseek 3Di strings.
2. Generate frozen SaProt embeddings from amino-acid plus 3Di tokens.
3. Generate frozen ESM3 embeddings from sequence and protein coordinates.
4. For each of five SSCP checkpoints, calculate residue logits and convert them to probabilities.
5. Average the five probability vectors residue by residue: `p_base = mean(sigmoid(z_fold_k), k=1..5)`.
6. Clip the mean probability to `[1e-6, 1-1e-6]` and calculate `z_base = logit(p_base)`.
7. Retrieve and filter remote structures with Foldseek, map aligned template residues to query residues and project training-memory labels.
8. Apply prior shrinkage and structural-confidence weighting to produce the remote residual `e_i`.
9. Calculate `z_final = z_base + alpha * e_i` and return `sigmoid(z_final)`. If there is no valid correspondence, set `e_i = 0`, giving exact fallback to SSCP.

## Frozen commands

After recreating the original directory contract and placing the Zenodo assets at paths declared in `audit/MODEL_ASSETS_FROZEN.json`, the provenance scripts are run in this order:

```bash
python reproducibility/frozen_pipeline/build_3di.py --root /path/to/run_root
python reproducibility/frozen_pipeline/generate_embeddings.py --root /path/to/run_root --kind saprot --device cuda:0 --scope full --domain dna
python reproducibility/frozen_pipeline/generate_embeddings.py --root /path/to/run_root --kind esm3 --device cuda:0 --scope full --domain dna
python reproducibility/frozen_pipeline/run_c0_prediction.py --root /path/to/run_root --domain dna --device cuda:0 --scope full
python reproducibility/frozen_pipeline/run_v1_memory_prediction.py --root /path/to/run_root --domain dna --scope full
```

Repeat the four domain-specific commands with `--domain rna`. A portable launcher that removes original server paths is listed as a release-blocking item in `docs/RELEASE_CHECKLIST.md`; until it is validated, do not describe this repository as a one-command package.

## ESM3 input boundary

The frozen script passes all fields returned by `model.encode(protein)` to ESM3. Because `ESMProtein.from_pdb()` is called without `with_annotations=True`, the encoded SS8, SASA, function and residue-annotation fields are `None`. ESM3 substitutes its PAD tokens for those optional tracks. The branch therefore uses sequence, structure tokens and coordinates and does not use external function, family or binding-site annotations.
