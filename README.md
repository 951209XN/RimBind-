# RIMBind

RIMBind predicts protein DNA and protein RNA binding residues by combining a query-intrinsic sequence-structure predictor with structure-aligned remote interface memory. Remote evidence is added as a bounded, confidence-weighted residual; when no valid residue correspondence is available, the method falls back exactly to the query-intrinsic prediction.

## Release status

This repository contains the audited public-release candidate for the manuscript. The source and reproducibility materials are now available on GitHub; the task-specific checkpoint archive has not yet been deposited in Zenodo. The repository contains the frozen model definition, the label-free inference pipeline used for the main predicted-monomer evaluation, protocol locks, checkpoint checksums, a denominator-level processed-data manifest and inference documentation. The task-specific checkpoint bundle and derived training-memory labels are intended for Zenodo rather than Git. All DOI placeholders must be replaced before submission.

## Audited implementation facts

- The final query-intrinsic model combines SaProt sequence-structure embeddings, a C-alpha geometric branch and frozen structure-aware ESM3 embeddings.
- Five fold outputs are converted to probabilities with `sigmoid` and then averaged. BayesRIM converts the mean probability back to a clipped logit before adding the remote residual.
- The ESM3 branch receives query sequence, structure tokens and protein coordinates. The audited call uses `ESMProtein.from_pdb()` with its default `with_annotations=False`; SS8, SASA, function and residue-annotation tracks are absent and are replaced by ESM3 default padding tokens. No database-derived function or binding annotations are supplied.
- The main historical evaluation uses ColabFold or AlphaFold2 predicted monomer structures. The temporal cohort uses protein chains extracted from experimental complexes after removing binding partners and is therefore a separately specified sensitivity cohort.

## Two reproduction entry points

### Quick inference

The intended input-output contract is `protein sequence + protein-only structure -> residue probability table`. The exact computation order is documented in `INFERENCE.md`. A portable launcher and a checksum-validated minimal example remain release gates and must not be claimed as complete until they pass on a clean machine.

### Paper reproduction

The frozen historical-test denominators are recorded in `data_manifests/`. Analysis reproduction should start from released frozen predictions/intermediate tables and regenerate the paired effects, bootstrap intervals, randomization summaries and figure-ready tables. Full retraining of SaProt or ESM3 is not required for statistical reproduction. Figure/statistics launchers remain a release gate until added and validated.

## Repository map

- `src/rimbind/model_single.py`: frozen SSCP model definition.
- `reproducibility/frozen_pipeline/`: source snapshot used by the audited main evaluation, with machine-specific paths replaced by explicit arguments or environment variables.
- `configs/`: main protocol and version lock.
- `checkpoints/manifest.json`: checkpoint, memory and external encoder checksums.
- `data_manifests/`: frozen historical-test sample manifest, denominator summary and checksums.
- `INFERENCE.md`: required inputs, environment and inference order.
- `CODE_AND_DATA_AVAILABILITY.md`: manuscript-ready availability statement.
- `THIRD_PARTY_ASSETS.md`: redistribution and license decision matrix.
- `SUBMISSION_STATEMENTS_TEMPLATE.md`: publication statements with author-supplied fields clearly marked.
- `docs/ESM3_TRACK_AUDIT_20261004.md`: code-level audit of optional ESM3 tracks.
- `docs/RELEASE_CHECKLIST.md`: remaining publication steps.

## Reproducibility boundary

SaProt and ESM3 foundation-model weights are not redistributed here. Obtain them from their official sources under their original licenses and verify the hashes in `checkpoints/manifest.json`. RCSB PDB entries remain available from RCSB. Predicted monomers can be regenerated from the released sequence manifests and frozen ColabFold protocol when those manifests are deposited.

## Citation

The manuscript citation and Zenodo DOI will be added when the public archive is released.
