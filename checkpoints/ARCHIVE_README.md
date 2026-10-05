# RIMBind Checkpoint Archive

The Zenodo archive contains ten task-specific SSCP checkpoints and two derived training-memory label files. Checkpoint names are normalized by task and fold; their SHA256 digests and original byte sizes are recorded in `manifest.json` and `SHA256SUMS`.

The archive does not include SaProt, ESM3 or Foldseek. Obtain those components from their official distributions and verify the hashes recorded in `manifest.json`.

The DNA and RNA checkpoint ensembles must be used separately. For each task, apply `sigmoid` to each fold's residue logits and calculate the arithmetic mean of the five probability vectors. Do not average logits.
