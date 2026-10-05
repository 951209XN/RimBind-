# Minimal core-inference example

The `6jhe_A` fixture is a 53-residue DNA-binding query represented by its amino-acid sequence and ColabFold/AlphaFold2 predicted monomer. It contains no benchmark labels.

After extracting `RIMBind_checkpoints_v1.tar.gz`, run:

```bash
python examples/run_core_smoke_test.py \
  --fixture examples/6jhe_A/core_inference_fixture.pt \
  --checkpoint-root /path/to/RIMBind_checkpoints_v1 \
  --model-code src/rimbind/model_single.py \
  --output /tmp/6jhe_A_prediction.csv \
  --device cuda:0
```

Expected output SHA256:

```text
729099fb7a43b1d2ec247d77e16a9e68ec3597d785cae69bc86ae512232ddf16
```

The smoke test executes all five frozen SSCP checkpoints, averages the five sigmoid probabilities, and applies the frozen DNA BayesRIM update. It reproduced `expected_output.csv` byte-for-byte on the audited server.

## Scope boundary

`core_inference_fixture.pt` contains frozen SaProt/ESM3 embeddings and Foldseek-derived remote evidence so that the model and fusion stages can be tested without redistributing third-party weights. This is a **core inference smoke test**, not yet a clean-machine test of the complete FASTA/PDB-to-embedding and structural-retrieval preprocessing chain. `example.fasta` and `example.pdb` are included as the public input reference for the future portable end-to-end launcher.
