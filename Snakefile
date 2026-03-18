# Snakefile
# Full pipeline for idr-sequence-predict

configfile: "config/config.yaml"

import os
from pathlib import Path

IDR = config["idr_mol_feats_repo"]
SEED = int(config.get("seed", 0))

WORK = "work"
FASTAS = f"{WORK}/fastas"
FEATS  = f"{WORK}/feats"
CTRL   = f"{WORK}/controls"
TRAIN  = f"{WORK}/train"
FI     = f"{WORK}/feature_importance"
MODELS = "models_all"
RESULTS = "results"
FSETS = "feature_sets"

POS_FASTA = config["positives_fasta"]
REF_FASTAS = config["reference_fastas"]
CONCAT_REF = bool(config.get("concat_reference", True))

WIN = config["window"]
WIN_SIZE = int(WIN.get("size", 15))
WIN_STEP = int(WIN.get("step", 1))
WIN_IDMODE = WIN.get("id_mode", "start")

FEATURE_MODE = config["feature_source"]["mode"]   # compute | use_res
POS_RES_IN = config["feature_source"].get("positives_feats", "")
REF_RES_IN = config["feature_source"].get("reference_feats", "")

PTM_IDS = config["ptm_ids"]
RANDOM_N = int(config["random_control_n"])
SHORT_MULTS = [int(x) for x in config.get("short_mults", [1,2,5,10])]

FI_SEED = int(config["feature_importance"].get("seed", 0))
CUM_THR = config["feature_importance"].get("cum_thresholds", [0.90, 0.95])

BETA = float(config.get("beta", 0.5))

# ---- Derived paths ----
REF_FASTA = f"{FASTAS}/reference.fasta"

POS_WIN = f"{FASTAS}/positives_w{WIN_SIZE}.fasta"
REF_WIN = f"{FASTAS}/reference_w{WIN_SIZE}.fasta"

POS_FEATS_RAW = f"{FEATS}/positives.f_FEAT.out.txt"
REF_FEATS_RAW = f"{FEATS}/reference.f_FEAT.out.txt"

POS_NORM = f"{FEATS}/positives.norm.tsv"
REF_NORM = f"{FEATS}/reference.norm.tsv"

PTM_POOL = f"{CTRL}/ptm_pool.tsv"
RAN_POOL = f"{CTRL}/random_pool.tsv"

def short_path(prefix, m):
    # User keeps these bucket files in inputs/
    # PTM:    inputs/ptms_short.txt, ptms_short2.txt, ptms_short5.txt, ptms_short10.txt
    # Random: inputs/random_short.txt, random_short2.txt, random_short5.txt, random_short10.txt
    suffix = "" if m == 1 else str(m)
    if prefix == "ptm":
        return f"inputs/ptms_short{suffix}.txt"
    elif prefix == "random":
        return f"inputs/random_short{suffix}.txt"
    else:
        raise ValueError(f"Unknown prefix: {prefix}")

PTM_SHORT = [short_path("ptm", m) for m in SHORT_MULTS]
RAN_SHORT = [short_path("random", m) for m in SHORT_MULTS]

def train_path(prefix, m):
    # Always include the multiplier in the filename to match wildcarded label rules
    # (so m=1 -> ..._short1.tsv)
    suffix = str(m)
    return f"{TRAIN}/{prefix}_short{suffix}.tsv"

TRAIN_PTM = [train_path("ptm", m) for m in SHORT_MULTS]
TRAIN_RAN = [train_path("random", m) for m in SHORT_MULTS]

FI_SCORES = f"{FI}/feature_importance_scores.csv"
FSET90 = f"{FSETS}/selected_feats_cum90.txt"
FSET95 = f"{FSETS}/selected_feats_cum95.txt"

# hyperparam cache + markers
HP_JSON = f"{MODELS}/hparams.json"
HP_LINEAR_DONE = f"{WORK}/hparams_linear.done"
HP_NN_DONE = f"{WORK}/hparams_nn.done"
HP_SM_BIG_DONE = f"{WORK}/hparams_smoteenn_big.done"
TRAIN_DONE = f"{WORK}/train_from_hparams.done"
PRED_DONE = f"{WORK}/predictions.done"

# consensus / merge / fasta outputs
CONS_DIR = f"{RESULTS}/consensus"
REG_DIR  = f"{RESULTS}/regions"
FA_DIR   = f"{RESULTS}/fasta"

THRESHOLDS = [100,95,90,85]

CONS_FILES = [f"{CONS_DIR}/frags__pct{p}.txt" for p in THRESHOLDS]
REG_FILES  = [f"{REG_DIR}/regions__pct{p}__min2frags.tsv" for p in THRESHOLDS]
FA_FILES   = [f"{FA_DIR}/regions__pct{p}__min2frags.fasta" for p in THRESHOLDS]

PARENT_FASTA_1 = "inputs/tf_modified_headers.fasta"
PARENT_FASTA_2 = "inputs/modified_headers_disordered_regions.fasta"


rule all:
    input:
        FSET90, FSET95,
        TRAIN_DONE,
        PRED_DONE,
        CONS_FILES,
        REG_FILES,
        FA_FILES


# ------------------------
# FASTA prep
# ------------------------
rule make_reference_fasta:
    input:
        REF_FASTAS
    output:
        REF_FASTA
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if CONCAT_REF:
            shell("python scripts/concat_fastas.py --fastas {input} --out {output}")
        else:
            if len(input) != 1:
                raise ValueError("concat_reference=false requires exactly one reference fasta")
            shell("cp {input[0]} {output}")


rule window_positives:
    input: POS_FASTA
    output: POS_WIN
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if WIN.get("enabled_positives", True):
            shell("python scripts/sliding_window.py --in-fasta {input} --out-fasta {output} --window {WIN_SIZE} --step {WIN_STEP} --id-mode {WIN_IDMODE}")
        else:
            shell("cp {input} {output}")


rule window_reference:
    input: REF_FASTA
    output: REF_WIN
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if WIN.get("enabled_reference", True):
            shell("python scripts/sliding_window.py --in-fasta {input} --out-fasta {output} --window {WIN_SIZE} --step {WIN_STEP} --id-mode {WIN_IDMODE}")
        else:
            shell("cp {input} {output}")


# ------------------------
# Features (idr.mol.feats)
# ------------------------
rule feats_positives:
    input: POS_WIN
    output: POS_FEATS_RAW
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if FEATURE_MODE == "use_res":
            if not POS_RES_IN:
                raise ValueError("feature_source.positives_feats must be set for mode=use_res")
            shell("cp {POS_RES_IN} {output}")
        else:
            shell("python scripts/prepare_idr_mol_feats_inputs.py --fastas {input} --data-dir {IDR}/DATA --list-file {IDR}/input_file_feats.txt --mode copy")
            shell("bash scripts/run_idr_mol_feats.sh {IDR} input_file_feats.txt")
            base = Path(str(input[0])).name
            src = f"{IDR}/RES/{base}.f_FEAT.out.txt"
            shell("cp {src} {output}")


rule feats_reference:
    input: REF_WIN
    output: REF_FEATS_RAW
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if FEATURE_MODE == "use_res":
            if not REF_RES_IN:
                raise ValueError("feature_source.reference_feats must be set for mode=use_res")
            shell("cp {REF_RES_IN} {output}")
        else:
            shell("python scripts/prepare_idr_mol_feats_inputs.py --fastas {input} --data-dir {IDR}/DATA --list-file {IDR}/input_file_feats.txt --mode copy")
            shell("bash scripts/run_idr_mol_feats.sh {IDR} input_file_feats.txt")
            base = Path(str(input[0])).name
            src = f"{IDR}/RES/{base}.f_FEAT.out.txt"
            shell("cp {src} {output}")


rule norm_pos:
    input: POS_FEATS_RAW
    output: POS_NORM
    shell: "python scripts/normal.py --infile {input} --outfile {output}"


rule norm_ref:
    input: REF_FEATS_RAW
    output: REF_NORM
    shell: "python scripts/normal.py --infile {input} --outfile {output}"


# ------------------------
# Controls
# ------------------------
rule ptm_pool:
    input:
        big=REF_NORM,
        ids=PTM_IDS
    output:
        PTM_POOL
    shell:
        "python scripts/subset_features_by_uniprot_ids.py --big-feats {input.big} --ids {input.ids} --out {output}"


rule random_pool:
    input:
        big=REF_NORM,
        ids=PTM_IDS
    output:
        RAN_POOL
    params:
        n=RANDOM_N,
        seed=SEED
    shell:
        "python scripts/sample_random_controls_from_big.py --big-feats {input.big} --out {output} --n {params.n} --exclude-ids {input.ids} --seed {params.seed}"


# ------------------------
# Label training tables
# ------------------------
rule label_train_ptm:
    input:
        pos=POS_NORM,
        ctrl=lambda wc: short_path("ptm", int(wc.mult))
    output:
        "work/train/ptm_short{mult}.tsv"
    wildcard_constraints:
        mult="1|2|5|10"
    shell:
        "python scripts/add_class_labels.py --positives {input.pos} --controls {input.ctrl} --out {output}"

rule label_train_random:
    input:
        pos=POS_NORM,
        ctrl=lambda wc: short_path("random", int(wc.mult))
    output:
        "work/train/random_short{mult}.tsv"
    wildcard_constraints:
        mult="1|2|5|10"
    shell:
        "python scripts/add_class_labels.py --positives {input.pos} --controls {input.ctrl} --out {output}"

# ------------------------
# Feature importance + cumulative feature sets
# ------------------------
rule feature_importance:
    input:
        train=train_path("ptm", 1)
    output:
        FI_SCORES
    params:
        seed=FI_SEED
    shell:
        "python scripts/run_feature_importance.py --train {input.train} --out {output} --seed {params.seed}"


rule select_feature_lists:
    input:
        FI_SCORES
    output:
        FSET90, FSET95
    params:
        thresholds=" ".join(str(x) for x in CUM_THR)
    shell:
        "python scripts/select_feature_lists.py --scores {input} --outdir {FSETS} --thresholds {params.thresholds}"


# ------------------------
# Hyperparam search (cached)
# ------------------------
rule search_linear_hparams:
    input:
        # ensure training tables exist first
        TRAIN_PTM + TRAIN_RAN
    output:
        HP_LINEAR_DONE
    shell:
        "python scripts/search_linear_hparams.py --data-dir inputs --out {HP_JSON} --beta {BETA} && touch {output}"


rule search_nn_hparams:
    input:
        TRAIN_PTM + TRAIN_RAN
    output:
        HP_NN_DONE
    shell:
        "python scripts/search_nn_hparams.py --data-dir inputs --out {HP_JSON} && touch {output}"


rule search_smoteenn_big_hparams:
    input:
        "inputs/final_PTM_no_tfs.txt",
        "inputs/final_random.txt"
    output:
        HP_SM_BIG_DONE
    shell:
        "python scripts/search_smoteenn_hparams.py --data-dir inputs --out {HP_JSON} && touch {output}"


# ------------------------
# Train from cached hparams
# ------------------------
rule train_from_hparams:
    input:
        HP_LINEAR_DONE, HP_NN_DONE, HP_SM_BIG_DONE
    output:
        TRAIN_DONE
    shell:
        "DATA_DIR=inputs HP={HP_JSON} ./scripts/run_train_from_hparams.sh && touch {output}"


# ------------------------
# Predict
# ------------------------
rule predict_all:
    input:
        TRAIN_DONE
    output:
        PRED_DONE
    shell:
        "FEATURES_FILE=inputs/final_all_noNaN.txt MODELS_ROOT={MODELS} ./scripts/run_predict_all.sh && touch {output}"


# ------------------------
# Consensus -> merge -> FASTA
# ------------------------
rule consensus_from_predictions:
    input:
        PRED_DONE
    output:
        CONS_FILES
    shell:
        "python scripts/consensus_from_predictions.py --pred-root results/predictions --outdir {CONS_DIR} --thresholds 100 95 90 85"


rule merge_regions:
    input:
        lambda wc: f"{CONS_DIR}/frags__pct{wc.pct}.txt"
    output:
        f"{REG_DIR}/regions__pct{{pct}}__min2frags.tsv"
    wildcard_constraints:
        pct="100|95|90|85"
    shell:
        "python scripts/merge_fragments_to_regions.py --frags {input} --window 15 --min-frags 2 --out {output}"

rule extract_fasta:
    input:
        regions=f"{REG_DIR}/regions__pct{{pct}}__min2frags.tsv",
        fa1=PARENT_FASTA_1,
        fa2=PARENT_FASTA_2
    output:
        f"{FA_DIR}/regions__pct{{pct}}__min2frags.fasta"
    wildcard_constraints:
        pct="100|95|90|85"
    shell:
        "python scripts/extract_regions_to_fasta_simple.py --regions {input.regions} --fasta {input.fa1} --fasta {input.fa2} --out {output}"