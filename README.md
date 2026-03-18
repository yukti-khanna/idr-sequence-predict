# idr-sequence-predict

[![Snakemake](https://img.shields.io/badge/snakemake-≥7-blue)](https://snakemake.readthedocs.io)
[![Conda](https://img.shields.io/badge/conda-environment-green)](environment.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A reproducible Snakemake pipeline to discover IDR sequence regions similar to a small input set (e.g., TAD-like fragments) across a reference proteome / IDR search space.**

This repository provides an end-to-end workflow to:

- (optionally) generate **sliding windows** from sequences,
- compute **sequence-derived features** via the published tool **`idr.mol.feats`**,
- build **control sets** (PTM-based and random),
- train multiple **ML classifiers** across feature sets and control regimes,
- generate proteome-wide **prediction lists**,
- compute **consensus** across model outputs,
- **merge overlapping fragments** into regions,
- export merged regions as **FASTA** for downstream analysis.

> **Primary use case:** You have a small set of “interesting” IDR sequences (e.g., known TADs) and want to find more *similar* sequence fragments across a large reference set (IDRs, TFs, or any proteome slice).

---

## Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Install](#install)
  - [Conda environment](#conda-environment)
  - [External dependency: idr.mol.feats](#external-dependency-idrmolfeats)
- [Quickstart](#quickstart)
- [Minimal example run](#minimal-example-run)
- [Configuration](#configuration)
  - [Feature source modes: compute vs use_res](#feature-source-modes-compute-vs-use_res)
  - [Sliding window options](#sliding-window-options)
  - [Controls](#controls)
  - [Reproducibility](#reproducibility)
- [Outputs](#outputs)
- [Run only part of the pipeline](#run-only-part-of-the-pipeline)
- [Example inputs](#example-inputs)
- [Troubleshooting](#troubleshooting)
- [Citations & attribution](#citations--attribution)
- [License](#license)

---

## Overview

This pipeline is designed for “**few known sequences → find more similar ones**” problems in IDR biology.

You provide:
- a **positive set** FASTA (e.g., known TADs), and
- a **reference search space** FASTA (e.g., all IDRs, optionally TF full-length).

The pipeline generates features per sequence/window, trains multiple models under multiple control regimes, predicts proteome-wide, and summarizes results as consensus regions.

---

## How it works

### High-level workflow

1. **Input positives**: `inputs/known_tads.fasta` (or any FASTA)
2. *(Optional)* **Sliding window** the positives (default: 15 aa, step=1)
3. **Reference search space**: one or more FASTAs (IDRs/TFs/anything)
4. *(Optional)* **Sliding window** the reference (can become very large)
5. **Feature extraction** using **`idr.mol.feats`** (external dependency)
6. **Normalization** (fill missing values consistently)
7. **Control pools**
   - PTM: subset by UniProt IDs
   - Random: sample N rows/windows from reference features
8. **Short control buckets** (1×, 2×, 5×, 10× positives size)
9. **Training / hyperparameter selection** across model families
10. **Proteome-wide prediction**
11. **Consensus** across prediction lists (100/95/90/85%)
12. **Merge fragments into regions** requiring ≥2 overlapping windows
13. **Export merged regions as FASTA**

### ID conventions

- Sliding-window headers are emitted as: `SEQ_ID:START` where `START` is **1-based**.
- Feature tables from `idr.mol.feats` include a `NAME` column that is carried through the pipeline.
- Control selection uses the **left side of `:`** (the `SEQ_ID`) as the UniProt-like identifier.

---

## Repository layout

```text
idr-sequence-predict/
├── Snakefile
├── environment.yml
├── config/
│   ├── config.yaml              # your local config (recommended gitignored)
│   └── config.yaml.example      # template config to copy
├── scripts/
│   ├── setup_external_deps.sh
│   ├── sliding_window.py
│   ├── concat_fastas.py
│   ├── prepare_idr_mol_feats_inputs.py
│   ├── run_idr_mol_feats.sh
│   ├── normal.py
│   ├── subset_features_by_uniprot_ids.py
│   ├── sample_random_controls_from_big.py
│   ├── make_short_buckets.py
│   ├── add_class_labels.py
│   ├── run_feature_importance.py
│   ├── select_feature_lists.py
│   ├── run_train_from_hparams.sh
│   ├── run_predict_all.sh
│   ├── predict_linear.py
│   ├── predict_nn.py
│   ├── predict_smoteenn.py
│   ├── consensus_from_predictions.py
│   ├── merge_fragments_to_regions.py
│   └── extract_regions_to_fasta_simple.py
├── inputs/
│   ├── *.example.fasta / *.example.txt   # small examples tracked
│   └── (your real datasets; typically not committed)
├── work/        # generated intermediates (ignored)
├── results/     # generated outputs (ignored)
└── external/    # external deps cloned here (ignored)
```

---

## Install

### Conda environment

Create the environment:

```bash
conda env create -f environment.yml
conda activate idrseq
snakemake --version
```

Update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate idrseq
```

### External dependency: idr.mol.feats

This pipeline uses the published feature generator:

- **idr.mol.feats**: https://github.com/IPritisanac/idr.mol.feats

We do **not** bundle it. Instead, run:

```bash
chmod +x scripts/setup_external_deps.sh
./scripts/setup_external_deps.sh
```

This clones the repo into `external/idr.mol.feats` (gitignored) and creates `DATA/` and `RES/` folders.

> `idr.mol.feats` expects input FASTAs in `DATA/` and writes feature tables to `RES/`. This repo provides wrappers that copy inputs into `DATA/` and pull outputs from `RES/`.

---

## Quickstart

1) Copy the template config:

```bash
cp config/config.yaml.example config/config.yaml
```

2) Edit `config/config.yaml` to point to your inputs.

3) Dry-run (recommended):

```bash
snakemake -n
```

4) Run with a chosen number of cores:

```bash
snakemake -j 4
```

## Minimal example run

This repository includes a tiny example dataset so you can validate the workflow quickly.

### 1) Create a config that points to the example inputs

Copy the template config and edit these fields:

```yaml
# config/config.yaml

# Example inputs shipped with the repo
positives_fasta: "inputs/known_tads.example.fasta"
reference_fastas:
  - "inputs/reference.example.fasta"
concat_reference: false

# Example PTM IDs shipped with the repo
ptm_ids: "inputs/PTM_list.example.txt"

# Keep the example run small
random_control_n: 2000

# Use compute mode (recommended). This requires external/idr.mol.feats
feature_source:
  mode: "compute"
  positives_feats: ""
  reference_feats: ""
```

> Tip: For the example run, you can set `window.enabled_reference: false` if you want an even smaller test.

### 2) Dry-run (recommended)

```bash
snakemake -n
```

### 3) Run a small target

To avoid running the entire pipeline, you can build only the feature lists:

```bash
snakemake -j 1 feature_sets/selected_feats_cum90.txt feature_sets/selected_feats_cum95.txt
```

If you want to test the full example end-to-end, run:

```bash
snakemake -j 2
```

---

## Configuration

All settings live in `config/config.yaml`.

### Feature source modes: compute vs use_res

| Mode | Description | When to use |
|------|-------------|-------------|
| `compute` (default) | Run `idr.mol.feats` to generate features from FASTA | Recommended / reproducible |
| `use_res` | Skip feature extraction and use precomputed `*.f_FEAT.out.txt` | Fast reruns / testing |

Example (compute):

```yaml
feature_source:
  mode: "compute"
  positives_feats: ""
  reference_feats: ""
```

Example (use_res):

```yaml
feature_source:
  mode: "use_res"
  positives_feats: "inputs/tads_15aa.f_FEAT.out.txt"
  reference_feats: "inputs/reference_15aa.f_FEAT.out.txt"
```

### Sliding window options

```yaml
window:
  enabled_positives: true
  enabled_reference: true
  size: 15
  step: 1
  id_mode: "start"
```

Notes:
- If `enabled_reference: true`, the reference can become **very large** (millions of windows).
- Default ID style is `SEQ_ID:START` with 1-based `START`.

### Controls

PTM control (UniProt IDs, one per line):

```yaml
ptm_ids: "inputs/PTM_list.txt"
```

Random control size:

```yaml
random_control_n: 200000
```

Short control buckets (multiples of positives size):

```yaml
short_mults: [1, 2, 5, 10]
```

### Reproducibility

```yaml
seed: 0
feature_importance:
  seed: 0
  cum_thresholds: [0.90, 0.95]
```

---

## Outputs

### Predictions

Stored under:
- `results/predictions/...`

### Consensus fragments

Stored under:
- `results/consensus/frags__pct100.txt`
- `results/consensus/frags__pct95.txt`
- `results/consensus/frags__pct90.txt`
- `results/consensus/frags__pct85.txt`

Consensus is computed directly from prediction lists (deduplicating within each list).

### Merged regions + FASTA

Merged regions (≥2 overlapping windows) are written to:
- `results/regions/regions__pct{100,95,90,85}__min2frags.tsv`

Extracted FASTA sequences:
- `results/fasta/regions__pct{100,95,90,85}__min2frags.fasta`

FASTA headers include coordinates:

```text
>SEQID:START-END|n_frags=...
```

---

## Run only part of the pipeline

Only feature sets (cum90/cum95):

```bash
snakemake -j 1 feature_sets/selected_feats_cum90.txt feature_sets/selected_feats_cum95.txt
```

Only consensus + merge + FASTA (if predictions exist):

```bash
snakemake -j 4 results/fasta/regions__pct95__min2frags.fasta
```

---

## Example inputs

This repo includes a tiny example dataset to validate the workflow without large downloads:

- `inputs/known_tads.example.fasta`
- `inputs/reference.example.fasta`
- `inputs/PTM_list.example.txt`

To run with examples, point your config to these example files.

---

## Troubleshooting

### Don’t run `python -m py_compile Snakefile`
A Snakefile contains Snakemake syntax (`rule all:`). Use `snakemake -n` instead.

### Feature extraction fails
- Ensure `external/idr.mol.feats` exists (run `./scripts/setup_external_deps.sh`).
- Ensure config points to `idr_mol_feats_repo: external/idr.mol.feats`.

### PTM subset is empty
Your PTM ID list must match reference headers (left of `:`). If `NAME` is `Q9XXX:123`, your PTM list must include `Q9XXX`.

---

## Citations & attribution

This pipeline calls an external published feature generator:

- `idr.mol.feats` repository: https://github.com/IPritisanac/idr.mol.feats

Please cite/acknowledge that work as appropriate in any downstream publication.

