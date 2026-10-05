# Zenodo upload checklist

## Do not upload until

- all authors approve the source-code license;
- all authors approve the task-specific checkpoint license;
- redistribution of derived training-memory labels is confirmed, or those files are replaced by permitted regeneration instructions;
- the GitHub `v1.0.0` tag and release commit are final;
- creator names, affiliations and ORCIDs are confirmed.

## Proposed archive contents

- `RIMBind_checkpoints_v1.tar.gz` (study-trained SSCP checkpoints and derived training memory only);
- GitHub source archive for the final `v1.0.0` tag;
- checkpoint manifest and SHA256 files;
- processed-data manifest and SHA256 files;
- release notes;
- environment specifications;
- Code/Data Availability statement;
- license files approved by the authors.

Do not include SaProt, ESM3, ColabFold/AlphaFold2 or Foldseek foundation weights/binaries.

## Metadata still required from authors

- Title: `RIMBind v1.0.0: code, checkpoints and reproducibility archive`
- Creators and ORCIDs: `TO_BE_COMPLETED`
- Description/abstract: use the final manuscript-approved software description.
- Keywords: protein–nucleic acid interaction; binding-site prediction; structural retrieval; remote interface memory.
- Related identifier: GitHub v1.0.0 release URL.
- License: must match the approved archive asset policy.
- Version: `v1.0.0`.

## After publication

1. Verify the Zenodo record and download links in a logged-out browser.
2. Record both the version DOI and concept DOI.
3. Insert the version DOI into GitHub README, release notes and manuscript.
4. Regenerate the GitHub release if DOI placeholders remain.
