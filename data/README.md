# Dataset acquisition, identity and attribution

No raw dataset images are redistributed in this repository.

| Source | Download / description | Used cohort |
|---|---|---|
| TCMP-300, Zhang et al., Scientific Data 2025 | [Paper](https://doi.org/10.1038/s41597-025-05522-7); [Figshare repository](https://doi.org/10.6084/m9.figshare.29432726) | 240 evaluation classes, 29,050 train, 11,993 query |
| Indian Medicinal Leaves Image Datasets, Pushpa B. R. / Shobha Rani | [Mendeley version 3](https://doi.org/10.17632/748f8jkphb.3); [DIMPSAR description](https://doi.org/10.1016/j.dib.2023.109388) | 60 leaf-folder classes, 3,955 train, 988 query |

The TCMP full public dataset has 300 classes and 52,089 images; the evaluation manifest is a subset rather than that entire release. For Indian data use **Medicinal Leaf dataset**, not the separate **Medicinal plant dataset** collection. Refer to each publisher's terms and CC BY 4.0 attribution; citations are supplied in `references.bib`.

## Expected image roots

```text
tcmp-300/
  train/<class-folder>/<image>
  val/<class-folder>/<image>
Medicinal Leaf dataset/
  <common-name-folder>/<image>
```

Use the exact `path` field in the manifests as the authority. It is relative to the dataset root; no account or server path is required. `scripts/verify_images.py` checks every selected image's byte SHA-256. The initial local acquisition did not record a complete archive revision; current public versions may differ. This release identifies the study files by path, dimensions/hash metadata, pixel hashes and similarity groups. A matching version label alone is not an image-identity check.

## Manifest schema

`manifests/{tcmp,indian}.json.gz` contains:

- `classes`: ordered class labels, defining integer class IDs.
- `train`, `query`: image records with stable `id`, relative `path`, label (`class` or `class_name`), SHA-256, group and available audit fields.
- `ids`: original frozen-feature row order.
- `plans`: support image IDs by `seed` and `shot`, with nested 4/16 shots.
- `roles`: integer known, pool-unknown and query-only class IDs by role seed.
- `data_root`: an environment placeholder; command-line `--data-root` supplies the actual local root.

Labels are retained for independent evaluation. Training/pseudo-labeling must not use concealed pool/query labels. `context_indices.zip` stores original training-index arrays for each capped contamination context. It contains no image pixels. Hashes and class labels retain their original scientific meaning; private server paths have been removed.
