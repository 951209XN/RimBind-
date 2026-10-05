# RIMBind v1.0.0 — manuscript release (DRAFT)

Do not create the final tag until the repository license, checkpoint redistribution decision and final release commit are approved.

## Scope

- DNA and RNA residue-level binding-site prediction.
- Query-intrinsic SSCP combining SaProt, C-alpha geometry and frozen structure-aware ESM3.
- Structure-aligned remote interface memory with bounded BayesRIM residual fusion.
- Exact SSCP fallback when no qualified remote correspondence is available.

## Audited implementation

- Five fold logits are independently transformed by sigmoid; the five probability vectors are averaged before clipping and logit conversion.
- ESM3 uses sequence, structure tokens and protein coordinates; optional SS8, SASA, function and residue-annotation tracks are absent/default padding.
- Historical benchmark evaluation uses predicted monomer structures.
- Complete DNA-181 reporting contains all 181 proteins and 75,258 residues.

## Reproducibility assets

- Frozen source and protocol locks.
- Checkpoint and data-manifest SHA256 files.
- Processed-data denominator manifest for 427 historical test proteins.
- `6jhe_A` label-free core-inference example with byte-level expected output.
- Separate checkpoint archive planned for Zenodo.

## Known release boundaries

- The full FASTA/PDB-to-prediction portable launcher remains to be clean-environment tested.
- SaProt, ESM3, ColabFold/AlphaFold2 and Foldseek weights/binaries are not redistributed.
- The temporal experimental-chain cohort uses a different structure-source protocol from the historical predicted-monomer benchmark.

## Finalization fields

- Final release commit: `TO_BE_INSERTED`
- Source-code license: `Apache-2.0`
- Checkpoint license: `AUTHOR_DECISION_REQUIRED`
- Zenodo DOI: `TO_BE_INSERTED`
- Manuscript citation: `TO_BE_INSERTED`
