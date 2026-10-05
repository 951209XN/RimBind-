# Asset and license matrix

This file separates study-owned assets from third-party dependencies. It is a release checklist, not legal advice. The exact upstream license terms must be verified at the release date.

| Asset | Source | Redistributed here? | Release treatment |
|---|---|---:|---|
| RIMBind/SSCP source code | This study | Planned | Add a repository license selected and approved by all authors. |
| Trained DNA/RNA SSCP checkpoints | This study | Zenodo planned | State an explicit weight-use license approved by all authors. |
| Derived DNA/RNA training-memory labels | This study, derived from benchmark labels | Zenodo planned | Confirm that redistribution is compatible with the source datasets; otherwise release regeneration scripts and accession/record pointers only. |
| SaProt source and foundation weights | Third party | No | Link to the official distribution and require users to obtain it under the upstream terms. Record the expected model identifier/hash. |
| ESM3 source and foundation weights | Third party | No | Link to the official distribution and require users to obtain it under the upstream terms. Record the expected model identifier/hash. |
| ColabFold/AlphaFold2 software and databases | Third party | No | Link to official installation/data instructions; publish only this study's frozen protocol and permitted derived structures/manifests. |
| Foldseek binary | Third party | No | Link to the official release and record the validated version/hash. |
| RCSB PDB structures and metadata | RCSB PDB | No bulk repackaging planned | Release accession lists and preprocessing provenance; users retrieve source records from RCSB. |

## Release rule

The checkpoint archive must contain only study-owned task-specific assets whose redistribution has been approved. It must not bundle SaProt, ESM3, ColabFold/AlphaFold2 or Foldseek binaries/weights. A top-level project license must not be added by automation without author approval because choosing it changes downstream reuse rights.
