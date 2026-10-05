# Processed-data manifest

`historical_test_processed_data_manifest.csv` is a denominator- and provenance-level manifest generated directly from the frozen server objects on 2026-10-05. It contains one row per historical external-test protein and does **not** contain residue labels, prediction probabilities or embeddings.

The manifest records protein ID, dataset, sequence/label length, positive and negative label counts, predicted-monomer structure provenance, structure SHA256, number of qualified remote hits, exact-fallback status and inference protocol. Its summary must agree with the manuscript:

| Dataset | Proteins | Residues | Positive residues | Remote evidence | Exact fallback |
|---|---:|---:|---:|---:|---:|
| DNA-129_Test | 129 | 37,515 | 2,240 | 103 | 26 |
| DNA-181_Test | 181 | 75,258 | 3,208 | 128 | 53 |
| RNA-117_Test | 117 | 37,345 | 2,031 | 77 | 40 |

The complete DNA-181 manifest combines the 163 proteins in the original frozen 0916 run with the 18 full-length long-sequence extensions. The latter used the same predicted-monomer input type and full-length model path; they were not filled with experimental-chain coordinates.

Run `sha256sum -c SHA256SUMS` from this directory to validate the two manifest files.
