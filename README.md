# RIMBind

RIMBind predicts protein DNA and protein RNA binding residues by combining a query-intrinsic sequence-structure predictor with structure-aligned remote interface memory. Remote evidence is added as a bounded, confidence-weighted residual; when no valid residue correspondence is available, the method falls back exactly to the query-intrinsic prediction.

## Release status

This repository is the audited public-release candidate for the manuscript. It contains the frozen model definition, the label-free inference pipeline used for the main predicted-monomer evaluation, protocol locks, checkpoint checksums, and an inference guide. The approximately 240 MB task-specific checkpoint bundle and derived training-memory labels are intended for Zenodo rather than Git. The Zenodo DOI placeholder must be replaced before manuscript submission.

## Audited implementation facts

- The final query-intrinsic model is the SaProt plus C-alpha geometry plus frozen ESM3 configuration recorded as variant C in the internal checkpoint metadata.
- Five fold outputs are converted to probabilities with `sigmoid` and then averaged. BayesRIM converts the mean probability back to a clipped logit before adding the remote residual.
- The ESM3 branch receives query sequence, structure tokens and protein coordinates. The audited call uses `ESMProtein.from_pdb()` with its default `with_annotations=False`; SS8, SASA, function and residue-annotation tracks are absent and are replaced by ESM3 default padding tokens. No database-derived function or binding annotations are supplied.
- The main historical evaluation uses ColabFold or AlphaFold2 predicted monomer structures. The temporal cohort uses protein chains extracted from experimental complexes after removing binding partners and is therefore a separately specified sensitivity cohort.

## Repository map

- `src/rimbind/model_single.py`: frozen SSCP model definition.
- `reproducibility/frozen_pipeline/`: source snapshot used by the audited 0916 main evaluation. The shell script retains its original server paths as provenance and is not a portable launcher.
- `configs/`: main protocol and version lock.
- `checkpoints/manifest.json`: checkpoint, memory and external encoder checksums.
- `INFERENCE.md`: required inputs, environment and inference order.
- `CODE_AND_DATA_AVAILABILITY.md`: manuscript-ready availability statement.
- `docs/ESM3_TRACK_AUDIT_20261004.md`: code-level audit of optional ESM3 tracks.
- `docs/RELEASE_CHECKLIST.md`: remaining publication steps.

## Reproducibility boundary

SaProt and ESM3 foundation-model weights are not redistributed here. Obtain them from their official sources under their original licenses and verify the hashes in `checkpoints/manifest.json`. RCSB PDB entries remain available from RCSB. Predicted monomers can be regenerated from the released sequence manifests and frozen ColabFold protocol when those manifests are deposited.

## Citation

The manuscript citation and Zenodo DOI will be added when the public archive is released.
