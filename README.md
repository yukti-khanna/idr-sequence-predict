# idr-sequence-predict  
**A reproducible Snakemake pipeline to discover IDR sequence regions similar to a small input set (e.g., TAD-like fragments) across a reference proteome / IDR search space.**

This repository provides an end-to-end workflow to:
- (optionally) generate **sliding windows** from sequences,
- compute **sequence-derived features** via an external published tool (**idr.mol.feats**),
- build **control sets** (PTM-based and random),
- train multiple **ML classifiers** across feature sets and control regimes,
- generate proteome-wide **prediction lists**,
- compute **consensus** across model outputs,
- **merge overlapping fragments** into regions,
- and export merged regions as **FASTA**.

---

## Quickstart

```bash
conda env create -f environment.yml
conda activate idrseq

chmod +x scripts/setup_external_deps.sh
./scripts/setup_external_deps.sh

cp config/config.yaml.example config/config.yaml
snakemake -n     # dry run
snakemake -j 4   # run
```

---

## External dependency

This pipeline uses `idr.mol.feats` for feature generation (not bundled here):
https://github.com/IPritisanac/idr.mol.feats

---

## Outputs

- `results/predictions/` — per-model prediction lists  
- `results/consensus/` — consensus fragments (pct100/95/90/85)  
- `results/regions/` — merged regions (≥2 overlapping frags)  
- `results/fasta/` — merged region sequences as FASTA  

---

## Notes

- Keep large outputs out of git (`work/`, `results/`, `external/` are gitignored).
- If you want to commit large inputs, use Git LFS.
