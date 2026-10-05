# Reproducibility status

## Verified

- Canonical model identity: SaProt + C-alpha geometry + frozen structure-aware ESM3.
- Five-fold aggregation: sigmoid per fold, arithmetic mean of probabilities, clipping to `[1e-6, 1-1e-6]`, then logit before BayesRIM.
- ESM3 information boundary: sequence, structure tokens and protein coordinates are populated; optional SS8, SASA, function and residue-annotation tracks are absent/default padding.
- Historical predicted-monomer denominators and exact-fallback counts: verified in `data_manifests/`.
- Task-specific checkpoint and training-memory file hashes: recorded in `checkpoints/`.
- The checkpoint archive was extracted independently on 2026-10-05 and all 12 task-specific files passed SHA256 verification; archive SHA256 is `93c7d2c063b8ee0f6f3829f074e7b2255f5a83197669376faca3c4ea4b323570`.
- All Python files in the release candidate passed syntax compilation, and the probability-mean-then-logit arithmetic was checked with a deterministic numerical example.

## Partially verified

- The provenance source snapshot matches the audited server logic and machine-specific model/data paths have been replaced by explicit arguments or environment variables. The portable path layer has passed static checks but not a clean-machine GPU run.
- Static checks and checksum verification can be run locally, but this is not equivalent to a clean-machine GPU reproduction.

## Not yet complete

- Portable one-command inference launcher.
- Minimal sequence/PDB example with expected probability-file checksum.
- Frozen-prediction-to-table/figure launchers.
- Clean-machine end-to-end execution.
- Public GitHub release tag.
- Zenodo archive and DOI.
- Author-approved source-code and checkpoint licenses.

These incomplete items are release-engineering tasks. They do not change the frozen scientific results, but the repository must not be described as fully reproducible until they are closed.
