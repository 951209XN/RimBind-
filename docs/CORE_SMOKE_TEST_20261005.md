# Public clean-checkout core smoke test

Date: 2026-10-05

Public repository commit tested:

```text
0653269e64404835351cb55d35027e57dad93afc
```

Procedure:

1. Clone the public GitHub repository into a new temporary directory.
2. Use the separately verified `RIMBind_checkpoints_v1` archive.
3. Run `examples/run_core_smoke_test.py` on the label-free `6jhe_A` fixture.
4. Compare the generated 53-residue CSV with the frozen expected output.

Audited environment:

```text
Python 3.10.19
PyTorch 2.1.1
NumPy 1.26.4
mamba-ssm 1.1.1
causal-conv1d 1.1.0
CUDA 11.8 environment
```

Result:

```text
CLEAN_CHECKOUT_CORE_SMOKE_PASS
SHA256 729099fb7a43b1d2ec247d77e16a9e68ec3597d785cae69bc86ae512232ddf16
```

The output matched `examples/6jhe_A/expected_output.csv` byte-for-byte.

## Boundary

This validates public-repository retrieval, checkpoint loading, five-fold probability aggregation and the frozen BayesRIM update. It reuses frozen label-free SaProt/ESM3 embeddings and remote evidence. It does not yet constitute a clean-environment execution of ColabFold, Foldseek, SaProt and ESM3 preprocessing from the raw FASTA/PDB inputs.
