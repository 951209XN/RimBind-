# Public Release Checklist

## Completed

- Final 0916 model identity and variant C model definition traced to frozen checkpoints.
- Probability-level five-fold aggregation traced in the frozen inference code.
- ESM3 optional-track audit completed; no function or residue-annotation leakage found.
- Ten task-specific checkpoint hashes and file sizes recorded.
- DNA and RNA training-memory hashes and file sizes recorded.
- Source snapshot, protocol lock, inference guide and manuscript availability text assembled.
- Historical external-test processed-data manifest generated from frozen labels and predictions; denominators and fallback counts verified.
- Supplementary Table S14 was separated into frozen endpoints and count-matched randomization sensitivity, with the real-minus-null estimand explicitly distinguished from RIMBind-versus-SSCP utility.

## Required before manuscript submission

- Select and add an explicit repository license.
- Obtain author confirmation for the task-specific checkpoint license and complete the third-party license matrix.
- Validate a portable launcher that replaces the original server absolute paths.
- Add a minimal example input and expected output checksum.
- Add and validate figure/statistics reproduction launchers against the frozen result tables.
- Upload the checkpoint and training-memory archive to Zenodo.
- Mint the Zenodo DOI and replace every placeholder in the manuscript and repository.
- Add the final manuscript author list and citation metadata.
- Test a clean installation on a machine that does not have access to the development server.

## Optional later work

- Add the temporal predicted-monomer matched sensitivity analysis. This is not part of the present release and the manuscript limitation regarding temporal structure-source mismatch must remain unless that analysis is completed.
