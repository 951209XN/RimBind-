# ESM3 Input Track Audit

## Conclusion

The audited RIMBind implementation does not supply ESM3 function annotations or residue-level annotations. It also does not calculate or supply DSSP SS8 or SASA on the normal PDB path. These four optional tracks are absent and are replaced inside ESM3 by default PAD tokens. The substantive ESM3 inputs are the query amino-acid sequence, structure tokens generated from the query protein coordinates, and the coordinates themselves. No function or binding-site annotation leakage was identified.

## Audited call chain

The main 0916 evaluation invokes the temporal-confirmation embedding script through `scripts/09_run_downstream_full.sh`. Lines 4 and 19-20 select `0824-BayesRIM-TemporalConfirmation-Frozen-V1/code/generate_embeddings.py` and the `xn_env_topjournal` ESM3 environment.

In `generate_embeddings.py`, lines 113-124 call `ESMProtein.from_pdb(..., chain_id="A")` and then `model.encode(protein)`. Lines 126-133 forward every encoded track without adding external annotations.

In the installed ESM package version 3.1.1, `esm/sdk/api.py` lines 62-76 show that `from_pdb()` calls `from_protein_chain()` without overriding `with_annotations=False`. Lines 78-95 explicitly set `secondary_structure=None`, `sasa=None` and `function_annotations=None` while retaining sequence and coordinates.

In `esm/models/esm3.py`, lines 425-430 initialize optional tracks to `None`; lines 438-447 only tokenize SS8 or SASA when present; and lines 476-489 only create function and residue-annotation tokens when `function_annotations` is present. Lines 329-344 of the forward path replace missing SS8, SASA, function and residue-annotation tracks with their corresponding PAD tokens.

## Provenance hashes

| File | SHA256 |
|---|---|
| `generate_embeddings.py` | `755f39e48b32261c40d3c9f03f74520d609a4836ad429984b80b43796c5c0d8e` |
| ESM 3.1.1 `sdk/api.py` | `be6d37847dc320894167fb7280c3f7293317783098099f3dae70410c278ef141` |
| ESM 3.1.1 `models/esm3.py` | `7e2a4c4d0a717b0a7c5767e7fc3a7ebfe2d670df16ed2f4e1e1b9b076c9d7c00` |

## Required manuscript correction

Do not state that RIMBind supplies SS8, SASA, function or residue-annotation values to ESM3. Use the following wording:

> Frozen ESM3 representations were generated from the query sequence and protein coordinates. `ESMProtein.from_pdb()` was called with its default `with_annotations=False`; consequently, no external function annotations, residue-level annotations, DSSP secondary-structure labels or SASA values were supplied. The ESM3 implementation filled those optional tracks with default padding tokens. Thus, the ESM3 branch conditioned on sequence and structure tokens and coordinates only and did not access database-derived functional or binding annotations.
