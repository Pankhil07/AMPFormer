#!/usr/bin/env python
import argparse
import random
import os
import csv
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from functools import partial
import re
from collections import defaultdict

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.dataset import DatasetConfig
from hyformer.models.auto import AutoModel
from hyformer.utils.tokenizers.auto import AutoTokenizer

# *** NOTE: now using V2 ***
from joint_improvement.generators import GumbeldoreMixinV2

# Optional wandb import (script still works without wandb if you don't use it)
try:
    import wandb
except ImportError:
    wandb = None

# --------------------------- label config ---------------------------
LABELS = [
    "A. baumannii ATCC 19606",                 # 00
    "E. coli ATCC 11775",                      # 01
    "E. coli AIC221",                          # 02
    "E. coli AIC222 - CRE",                    # 03
    "K. pneumoniae ATCC 13883",                # 04
    "P. aeruginosa PAO1",                      # 05
    "P. aeruginosa PA14",                      # 06
    "S. aureus ATCC 12600",                    # 07
    "S. aureus ATCC BAA-1556 - MRSA",          # 08
    "E. faecalis ATCC 700802 - VRE",           # 09
    "E. faecium ATCC 700221 - VRE",            # 10
]
assert len(LABELS) == 11

GNEG_IDXS = [0, 1, 2, 3, 4, 5, 6]
GPOS_IDXS = [7, 8, 9, 10]

# ============================================================
#  BASIC VALIDITY
# ============================================================

def is_valid_peptide(seq: str, min_len: int = 2, max_len: int = 50) -> bool:
    if not seq or not (min_len <= len(seq) <= max_len):
        return False
    return all(ch in "ACDEFGHIKLMNPQRSTVWY" for ch in seq)


def set_global_seed(seed: int = 1245):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def load_model_tokenizer(
    tokenizer_config_path: str,
    model_config_path: str,
    dataset_config_path: str,
    ckpt_path: str,
    device: torch.device,
):
    """Load Hyformer the same way as in your MIC generation script."""
    tok_cfg = TokenizerConfig.from_config_filepath(tokenizer_config_path)
    mdl_cfg = ModelConfig.from_config_filepath(model_config_path)
    dset_cfg = DatasetConfig.from_config_filepath(dataset_config_path)

    tokenizer = AutoTokenizer.from_config(tok_cfg)
    model = AutoModel.from_config(
        mdl_cfg,
        prediction_task_type=dset_cfg.prediction_task_type,
        num_prediction_tasks=dset_cfg.num_prediction_tasks,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_pretrained(state_dict=ckpt["model"], discard_prediction_head=False)
    model.to(device)
    model.eval()  # make sure dropout etc. are off
    return model, tokenizer, dset_cfg


def to_prediction_inputs(gen_ids: torch.Tensor, pred_task_token_id: int) -> torch.Tensor:
    pred_ids = gen_ids.clone()
    pred_ids[:, 0] = pred_task_token_id
    return pred_ids

# ============================================================
#  SEQUENCE PROPERTY HELPERS (no pandas)
# ============================================================

def calculate_hydrophobic_hydrophilic_residue_balance(sequence: str) -> float:
    hydrophobic_residues = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'Y'}
    hydrophilic_residues = {'R', 'N', 'D', 'Q', 'E', 'H', 'K', 'S', 'T', 'C'}

    hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_residues)
    hydrophilic_count = sum(1 for aa in sequence if aa in hydrophilic_residues)

    total_residues = len(sequence)
    if total_residues == 0:
        return 0.0

    balance = abs(hydrophobic_count - hydrophilic_count) / total_residues
    return balance


def count_consecutive_glycine_repeats(sequence: str, max_consecutive_glycine: int = 2) -> int:
    glycine_repeats = re.findall(r'(G+)', sequence)
    num_repeats = 0
    for repeat in glycine_repeats:
        length = len(repeat)
        if length > max_consecutive_glycine:
            num_repeats += 1
    return num_repeats


def count_consecutive_bulky_aa_repeats(sequence: str, max_consecutive_repeats: int = 2) -> int:
    bulky_repeats = re.findall(r'([WYF]+)', sequence)
    num_repeats = 0
    for repeat in bulky_repeats:
        length = len(repeat)
        if length > max_consecutive_repeats:
            num_repeats += 1
    return num_repeats


def check_motif_presence(sequence: str, motifs: list = None) -> bool:
    AGGREGATION_PRONE_MOTIFS = [
        'VLVL', 'WWYF', 'KLLL', 'STVIIE', 'VQIVYK', 'GNNQQNY', 'QYNNQ',
        'QQQQQQQQQ', 'NNNNNNNN', 'GSVIIE', 'VQIVYK', 'SSTY', 'SVQIVY',
        'KLVFFA', 'LVFFAEDVGSNK', 'VTGVTAVAQK', 'GSSGSS', 'VLIVLG',
        'YVIVFV', 'LIVVV', 'WLIVI', 'VVVIV', 'LLVVV', 'PXXPXX', 'PQPQPQ',
        'LVFFA', 'IYFV', 'STVIIE', 'TTVIE', 'NFGAIL', 'DFNKF'
    ]
    motifs = AGGREGATION_PRONE_MOTIFS if motifs is None else motifs
    return any(motif in sequence for motif in motifs)


def calculate_proline_content(sequence: str) -> float:
    if not sequence:
        return 0.0
    proline_count = sequence.count('P')
    total_residues = len(sequence)
    return proline_count / total_residues


def calculate_net_positive_charge(sequence: str) -> int:
    positively_charged = {'R', 'K', 'H'}
    negatively_charged = {'D', 'E'}

    positive_count = sum(sequence.count(aa) for aa in positively_charged)
    negative_count = sum(sequence.count(aa) for aa in negatively_charged)
    net_charge = positive_count - negative_count
    return net_charge


def calculate_cationic_content(sequence: str) -> int:
    if not sequence:
        return 0
    cationic_residues = {'R', 'K', 'H'}
    cationic_count = sum(sequence.count(aa) for aa in cationic_residues)
    return cationic_count


def has_repeated_tripeptide(sequence: str) -> bool:
    tripeptide_counts = defaultdict(int)
    seq_length = len(sequence)
    for i in range(seq_length - 2):
        tripeptide = sequence[i:i+3]
        tripeptide_counts[tripeptide] += 1
        if tripeptide_counts[tripeptide] > 3:
            return True
    return False


def hydrophilic_residue_abundance(sequence: str) -> float:
    hydrophilic_residues = set("KRHDESTNQY")
    hydrophilic_count = sum(1 for aa in sequence if aa in hydrophilic_residues)
    total_residues = len(sequence)
    if total_residues == 0:
        return 0.0
    abundance = (hydrophilic_count / total_residues) * 100
    return round(abundance, 2)


def max_consecutive_glycine(sequence: str) -> int:
    max_count = 0
    current_count = 0
    for aa in sequence:
        if aa == 'G':
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


def count_bulky_amino_acids(sequence: str) -> int:
    bulky_residues = set("FYWILVMRH")
    global_max = 0
    for residue in bulky_residues:
        max_count = 0
        current_count = 0
        for aa in sequence:
            if aa == residue:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        if max_count >= global_max:
            global_max = max_count
    return global_max


def detect_aggregation_prone_motifs(sequence: str) -> int:
    aggregation_motifs = [
        "VLVL", "WWYF", "KLLL", "STVIIE", "VQIVYK", "SSTY", "SVQIVY",
        "KLVFFA", "LVFFAEDVGSNK", "VTGVTAVAQK", "GSSGSS", "VLIVLG",
        "YVIVFV", "LIVVV", "WLIVI", "VVVIV", "WLIVI", "VVV",
        "PXXPXX", "PQPQPQ", "LVFFA", "IYFV", "STVIIE",
        "TTVIE", "NFGAIL", "DFNKF"
    ]
    for motif in aggregation_motifs:
        if motif in sequence:
            return 1
    return 0


def proline_residue_rate(sequence: str) -> float:
    total_residues = len(sequence)
    if total_residues == 0:
        return 0.0
    proline_count = sequence.count('P')
    rate = (proline_count / total_residues) * 100
    return round(rate, 2)


def count_polymerizing_cysteines(sequence: str) -> int:
    cysteine_count = sequence.count('C')
    polymerizing_cysteines = (cysteine_count // 2) * 2
    return polymerizing_cysteines


# pKa values of ionizable groups
pKa_values = {
    'N_term': 9.0,
    'C_term': 3.1,
    'D': 3.9,
    'E': 4.3,
    'C': 8.3,
    'Y': 10.1,
    'H': 6.0,
    'K': 10.5,
    'R': 12.5
}


def henderson_hasselbalch(pKa, pH, is_acidic):
    ratio = 10 ** (pKa - pH) if is_acidic else 10 ** (pH - pKa)
    return -1 / (1 + ratio) if is_acidic else 1 / (1 + ratio)


def calculate_charge(sequence, pH=7.0):
    charge = 0
    charge += henderson_hasselbalch(pKa_values['N_term'], pH, is_acidic=False)
    charge += henderson_hasselbalch(pKa_values['C_term'], pH, is_acidic=True)
    for aa in sequence:
        if aa in pKa_values:
            is_acidic = aa in "DECY"
            charge += henderson_hasselbalch(pKa_values[aa], pH, is_acidic)
    return round(charge, 2)

# ============================================================
#  RULE CHECKS ON A SINGLE SEQUENCE
# ============================================================

def is_peptide_ok_seq(seq: str) -> bool:
    """Port of is_peptide_ok but operating directly on a sequence string."""
    if len(seq) < 8:
        return False
    if len(seq) > 50:
        return False

    hh_balance = calculate_hydrophobic_hydrophilic_residue_balance(seq)
    if hh_balance < 0.3:
        return False
    if hh_balance > 0.7:
        return False

    gly_repeats = count_consecutive_glycine_repeats(seq)
    if gly_repeats > 0:
        return False

    bulky_repeats = count_consecutive_bulky_aa_repeats(seq)
    if bulky_repeats > 1:
        return False

    if check_motif_presence(seq):
        return False

    proline_content = calculate_proline_content(seq)  # fraction
    if proline_content > 0.2:
        return False

    net_charge = calculate_net_positive_charge(seq)
    if net_charge < 2:
        return False
    if net_charge > 10:
        return False

    cationic_content = calculate_cationic_content(seq)
    if cationic_content > 10:
        return False

    if has_repeated_tripeptide(seq):
        return False

    return True


def is_peptide_probable_seq(seq: str) -> bool:
    """Port of is_peptide_probable but operating directly on a sequence string."""
    hyd_ab = hydrophilic_residue_abundance(seq)
    if hyd_ab < 30 or hyd_ab > 70:
        return False

    if max_consecutive_glycine(seq) > 2:
        return False

    if count_bulky_amino_acids(seq) > 1:
        return False

    if detect_aggregation_prone_motifs(seq) == 1:
        return False

    if proline_residue_rate(seq) > 20:
        return False

    if count_polymerizing_cysteines(seq) > 1:
        return False

    charge = calculate_charge(seq)
    if charge > 10 or charge < 1.9:
        return False

    return True

# ============================================================
#  Levenshtein + diversity
# ============================================================

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    if len_b > len_a:
        a, b = b, a
        len_a, len_b = len_b, len_a

    prev_row = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr_row = [i]
        ca = a[i - 1]
        for j in range(1, len_b + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(
                    prev_row[j] + 1,
                    curr_row[j - 1] + 1,
                    prev_row[j - 1] + cost,
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def report_diversity_per_round(
    round_to_seqs: Dict[int, List[str]],
    max_pairs_per_round: int = 2000,
    out_csv: str = None,
    wandb_run=None,
):
    rows = []
    print("\nPer-round diversity (Levenshtein, possibly subsampled):")

    wb_table = None
    if wandb_run is not None and wandb is not None:
        wb_table = wandb.Table(
            columns=[
                "round", "n_seqs", "total_pairs", "pairs_used",
                "mean_lev", "mean_lev_norm"
            ]
        )

    for r in sorted(round_to_seqs.keys()):
        seqs = round_to_seqs[r]
        n = len(seqs)
        total_pairs = n * (n - 1) // 2

        if n < 2:
            print(f"  Round {r}: n={n} (too few sequences for diversity, set to 0)")
            rows.append([r, n, total_pairs, 0, 0.0, 0.0])

            if wb_table is not None:
                wb_table.add_data(r, n, total_pairs, 0, 0.0, 0.0)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "diversity/n_seqs": n,
                        "diversity/total_pairs": total_pairs,
                        "diversity/pairs_used": 0,
                        "diversity/mean_lev": 0.0,
                        "diversity/mean_lev_norm": 0.0,
                        "round": r,
                    },
                    step=r,
                )
            continue

        raw_dists = []
        norm_dists = []

        if total_pairs <= max_pairs_per_round:
            effective_pairs = total_pairs
            for i in range(n):
                si = seqs[i]
                for j in range(i + 1, n):
                    sj = seqs[j]
                    d = levenshtein(si, sj)
                    raw_dists.append(d)
                    L = max(len(si), len(sj))
                    norm_dists.append(d / L if L > 0 else 0.0)
        else:
            effective_pairs = max_pairs_per_round
            for _ in range(max_pairs_per_round):
                i = random.randrange(n)
                j = random.randrange(n - 1)
                if j >= i:
                    j += 1
                if j < i:
                    i, j = j, i
                si = seqs[i]
                sj = seqs[j]
                d = levenshtein(si, sj)
                raw_dists.append(d)
                L = max(len(si), len(sj))
                norm_dists.append(d / L if L > 0 else 0.0)

        mean_raw = float(np.mean(raw_dists))
        mean_norm = float(np.mean(norm_dists))
        print(
            f"  Round {r}: n={n}, pairs_used={effective_pairs}/{total_pairs}, "
            f"mean_lev={mean_raw:.3f}, mean_lev_norm={mean_norm:.3f}"
        )

        rows.append([r, n, total_pairs, effective_pairs, mean_raw, mean_norm])

        if wandb_run is not None:
            wandb_run.log(
                {
                    "diversity/n_seqs": n,
                    "diversity/total_pairs": total_pairs,
                    "diversity/pairs_used": effective_pairs,
                    "diversity/mean_lev": mean_raw,
                    "diversity/mean_lev_norm": mean_norm,
                    "round": r,
                },
                step=r,
            )
        if wb_table is not None:
            wb_table.add_data(r, n, total_pairs, effective_pairs, mean_raw, mean_norm)

    if wandb_run is not None and wb_table is not None:
        wandb_run.log({"diversity/table": wb_table})

    if out_csv is not None:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["round", "n_seqs", "total_pairs", "pairs_used",
                 "mean_lev", "mean_lev_norm"]
            )
            w.writerows(rows)
        print(f"[+] Wrote diversity CSV to: {out_csv}")


def plot_diversity_by_round(
    round_to_seqs: Dict[int, List[str]],
    max_pairs_per_round: int,
    out_png: str,
    title: str = "Diversity per Round (mean Levenshtein)",
):
    if out_png is None:
        print("[!] No out_png provided for diversity plot, skipping.")
        return

    rounds = sorted(round_to_seqs.keys())
    if not rounds:
        print("[!] No sequences to plot diversity.")
        return

    mean_lev = []
    mean_lev_norm = []

    for r in rounds:
        seqs = round_to_seqs[r]
        n = len(seqs)
        if n < 2:
            mean_lev.append(0.0)
            mean_lev_norm.append(0.0)
            continue

        total_pairs = n * (n - 1) // 2
        raw_dists = []
        norm_dists = []

        if total_pairs <= max_pairs_per_round:
            for i in range(n):
                si = seqs[i]
                for j in range(i + 1, n):
                    sj = seqs[j]
                    d = levenshtein(si, sj)
                    raw_dists.append(d)
                    L = max(len(si), len(sj))
                    norm_dists.append(d / L if L > 0 else 0.0)
        else:
            for _ in range(max_pairs_per_round):
                i = random.randrange(n)
                j = random.randrange(n - 1)
                if j >= i:
                    j += 1
                if j < i:
                    i, j = j, i
                si = seqs[i]
                sj = seqs[j]
                d = levenshtein(si, sj)
                raw_dists.append(d)
                L = max(len(si), len(sj))
                norm_dists.append(d / L if L > 0 else 0.0)

        mean_lev.append(float(np.mean(raw_dists)))
        mean_lev_norm.append(float(np.mean(norm_dists)))

    x = np.arange(len(rounds))

    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(x, mean_lev, marker="o", label="mean_lev")
    ax.plot(x, mean_lev_norm, marker="s", label="mean_lev_norm")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in rounds], rotation=30, ha="right")
    ax.set_xlabel("Round")
    ax.set_ylabel("Diversity")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[+] Saved diversity per-round plot to: {out_png}")

# ============================================================
#  Patch Gumbeldore V2 onto model
# ============================================================

def patch_gumbeldore_mixin(model):
    """
    Attach all relevant methods from GumbeldoreMixinV2 onto an existing model
    instance, so it behaves as if it inherited from the mixin.
    """
    for name in dir(GumbeldoreMixinV2):
        if name.startswith("__"):
            continue
        attr = getattr(GumbeldoreMixinV2, name)
        if not callable(attr):
            continue
        if name == "generate" or not hasattr(model, name):
            bound = attr.__get__(model, model.__class__)
            setattr(model, name, bound)


def patch_get_model_logits(model: torch.nn.Module):
    """
    Override _get_model_logits so that it always returns shape (batch, 1, vocab),
    which is what Gumbeldore expects.
    """

    @torch.inference_mode()
    def _get_model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Use LM head; build a simple attention mask
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        out = self.forward(
            input_ids=input_ids,
            attention_mask=None,
            next_token_only=True,
            task="lm",
        )

        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out

        # Ensure shape (batch, 1, vocab)
        if logits.dim() == 2:
            # (B, V) -> (B, 1, V)
            logits = logits.unsqueeze(1)
        elif logits.dim() == 3:
            # (B, T, V) -> (B, 1, V) (take last token if T>1)
            if logits.size(1) != 1:
                logits = logits[:, -1:, :]
        else:
            raise RuntimeError(f"Unexpected logits shape: {logits.shape}")

        return logits

    model._get_model_logits = _get_model_logits.__get__(model, model.__class__)

# small helper for prediction using V2's _get_model_predictions
def _predict_mic_vec(model, pred_ids, attn_mask):
    """
    Use GumbeldoreMixinV2._get_model_predictions to get MIC vector (numpy).
    pred_ids: (1, L)
    attn_mask: (1, L) bool/long
    """
    with torch.inference_mode():
        logits = model._get_model_predictions(
            input_ids=pred_ids,
            attention_mask=attn_mask
        )  # shape (1, num_tasks)
    mic_vec = logits[0].detach().cpu().numpy()
    return mic_vec

# ============================================================
#  MIC oracle WITH INTEGRATED SCREENING RULES (using V2)
# ============================================================

def peptide_mic_oracle_fn(
    input_idx: torch.Tensor,
    model,
    tokenizer,
    pred_token_id: int,
    pad_id: int,
    device: torch.device,
    alpha_gneg: float = 1.0,
    beta_gpos: float = 1.0,
    baseline_gneg: float = 3.0,
    baseline_gpos: float = 3.0,
    min_len: int = 2,
    max_len: int = 50,
    penalty_ok: float = 20.0,
    penalty_probable: float = 20.0,
) -> float:
    """
    Selectivity oracle with built-in sequence screening:

      - Base reward: reward = gneg / gpos
      - Screening rules:
          * is_peptide_ok_seq(seq)
          * is_peptide_probable_seq(seq)
      - Violations reduce the reward via penalties (not post-hoc filtering).

    Notes:
      * We still hard-reject obvious invalid peptides (charset/length)
        by returning a very low reward.
    """
    seq = tokenizer.decode(input_idx.tolist(), skip_special_tokens=True).strip().upper()

    # Basic validity: charset + length
    if not is_valid_peptide(seq, min_len=min_len, max_len=max_len):
        return -10.0  # strongly discouraged, but finite

    # MIC prediction using V2 helper
    input_ids = input_idx.unsqueeze(0).to(device)  # (1, L)
    pred_ids = to_prediction_inputs(input_ids, pred_token_id)
    attn_mask = (pred_ids != pad_id)

    mic_vec = _predict_mic_vec(model, pred_ids, attn_mask)

    gneg = float(mic_vec[GNEG_IDXS].mean())
    gpos = float(mic_vec[GPOS_IDXS].mean())

    if not (np.isfinite(gneg) and np.isfinite(gpos)) or gpos == 0.0:
        return -10.0

    # base reward: want gneg high, gpos low
    base_reward = gneg / gpos

    # Apply screening penalties
    penalty = 0.0
    if not is_peptide_ok_seq(seq):
        penalty += penalty_ok
    if not is_peptide_probable_seq(seq):
        penalty += penalty_probable

    reward = base_reward - penalty

    # If you want some lower bound to avoid extreme negatives:
    # min_reward = -10.0
    # reward = float(max(reward, min_reward))

    return float(reward)


def make_advantage_fn(adv_scale: float):
    def advantage_fn(reward: float) -> float:
        return adv_scale * reward
    return advantage_fn

# ============================================================
#  Plotting Gram-/Gram+
# ============================================================

def plot_gneg_gpos_by_round(
    round_to_gneg: Dict[int, List[float]],
    round_to_gpos: Dict[int, List[float]],
    out_png: str,
    title: str = "Gram− vs Gram+ pMIC by Round (Gumbeldore-guided)",
):
    if out_png is None:
        print("[!] No out_png provided, skipping plot.")
        return

    round_numbers = sorted(set(round_to_gneg.keys()) | set(round_to_gpos.keys()))
    if not round_numbers:
        print("No values to plot.")
        return

    gneg_values = [np.array(round_to_gneg.get(r, []), dtype=float) for r in round_numbers]
    gpos_values = [np.array(round_to_gpos.get(r, []), dtype=float) for r in round_numbers]

    valid_rounds, gneg_plot, gpos_plot = [], [], []
    for r, gn, gp in zip(round_numbers, gneg_values, gpos_values):
        if len(gn) == 0 and len(gp) == 0:
            continue
        valid_rounds.append(r)
        gneg_plot.append(gn if len(gn) > 0 else np.array([np.nan]))
        gpos_plot.append(gp if len(gp) > 0 else np.array([np.nan]))

    x = np.arange(len(valid_rounds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    pos_gneg = x - width / 2
    pos_gpos = x + width / 2

    bp_gneg = ax.boxplot(
        gneg_plot,
        positions=pos_gneg,
        widths=width,
        patch_artist=True,
        showmeans=True,
        manage_ticks=False,
    )
    bp_gpos = ax.boxplot(
        gpos_plot,
        positions=pos_gpos,
        widths=width,
        patch_artist=True,
        showmeans=True,
        manage_ticks=False,
    )

    gneg_color = "#4C78A8"
    gpos_color = "#F58518"

    for patch in bp_gneg["boxes"]:
        patch.set_facecolor(gneg_color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(gneg_color)

    for patch in bp_gpos["boxes"]:
        patch.set_facecolor(gpos_color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(gpos_color)

    for med in bp_gneg["medians"]:
        med.set_color(gneg_color); med.set_linewidth(2)
    for med in bp_gpos["medians"]:
        med.set_color(gpos_color); med.set_linewidth(2)

    for mean in bp_gneg["means"]:
        mean.set_markerfacecolor(gneg_color)
        mean.set_markeredgecolor(gneg_color)
    for mean in bp_gpos["means"]:
        mean.set_markerfacecolor(gpos_color)
        mean.set_markeredgecolor(gpos_color)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in valid_rounds], rotation=30, ha="right")
    ax.set_xlabel("Round")
    ax.set_ylabel("pMIC -(log10(MIC / 1 M))")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=gneg_color, edgecolor=gneg_color, alpha=0.6, label="Gram− mean pMIC"),
            Patch(facecolor=gpos_color, edgecolor=gpos_color, alpha=0.6, label="Gram+ mean pMIC"),
        ],
        loc="best"
    )

    gneg_means = np.array([np.nanmean(v) for v in gneg_plot])
    gpos_means = np.array([np.nanmean(v) for v in gpos_plot])

    if len(x) >= 2:
        coeff_gneg = np.polyfit(x, gneg_means, 1)
        ax.plot(x, np.poly1d(coeff_gneg)(x), "--", linewidth=2, color=gneg_color)
        ax.text(
            x[-1] + 0.2, np.poly1d(coeff_gneg)(x)[-1],
            f"β_gneg={coeff_gneg[0]:.4f}", fontsize=10, fontweight="bold", color=gneg_color
        )

        coeff_gpos = np.polyfit(x, gpos_means, 1)
        ax.plot(x, np.poly1d(coeff_gpos)(x), "--", linewidth=2, color=gpos_color)
        ax.text(
            x[-1] + 0.2, np.poly1d(coeff_gpos)(x)[-1] - 0.1,
            f"β_gpos={coeff_gpos[0]:.4f}", fontsize=10, fontweight="bold", color=gpos_color
        )

    plt.tight_layout()

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[+] Saved Gram−/Gram+ per-round plot to: {out_png}")

# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Selective Gumbeldore MIC-guided Hyformer + rule-aware oracle + plots."
    )
    parser.add_argument("--tokenizer_config_path", required=True)
    parser.add_argument("--model_config_path", required=True)
    parser.add_argument("--dataset_config_path", required=True)
    parser.add_argument("--ckpt_path", required=True)

    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--max_sequence_length", type=int, default=50)
    parser.add_argument("--advantage_constant", type=float, default=1.0)
    parser.add_argument("--min_nucleus_top_p", type=float, default=1.0)

    parser.add_argument("--alpha_gneg", type=float, default=1.0)
    parser.add_argument("--beta_gpos", type=float, default=1.0)
    parser.add_argument("--baseline_gneg", type=float, default=3.0)
    parser.add_argument("--baseline_gpos", type=float, default=3.0)

    parser.add_argument("--adv_scale", type=float, default=0.25)

    parser.add_argument("--out_plot", type=str, default=None)
    parser.add_argument("--out_csv", type=str, default=None)

    parser.add_argument(
        "--max_div_pairs_per_round",
        type=int,
        default=2000,
        help="Max Levenshtein pairs per round to estimate diversity (subsample if more).",
    )
    parser.add_argument(
        "--out_diversity_csv",
        type=str,
        default=None,
        help="Path to CSV file for per-round diversity statistics.",
    )

    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=105)

    # ---- W&B options ----
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)

    args = parser.parse_args()

    base_dir = "/p/project1/hai_1057/pankhil/hyformer/results_gumbeldore"
    if args.out_csv is None:
        args.out_csv = os.path.join(
            base_dir,
            f"sequences_adv_const_{args.advantage_constant}_top_p_{args.min_nucleus_top_p}_seed_{args.seed}_rounds{args.num_rounds}_bw_{args.beam_width}.csv",
        )
    if args.out_plot is None:
        args.out_plot = os.path.join(
            base_dir,
            f"sequences_adv_const_{args.advantage_constant}_top_p_{args.min_nucleus_top_p}_seed_{args.seed}.png",
        )
    if args.out_diversity_csv is None:
        args.out_diversity_csv = os.path.join(
            base_dir,
            f"diversity_adv_const_{args.advantage_constant}_top_p_{args.min_nucleus_top_p}_seed_{args.seed}.csv",
        )

    # ----------------- W&B init -----------------
    wandb_run = None
    if args.wandb_project is not None and wandb is not None:
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode

        wandb_kwargs = {
            "project": args.wandb_project,
            "config": vars(args),
        }
        if args.wandb_entity is not None:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name is not None:
            wandb_kwargs["name"] = args.wandb_run_name
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        if args.wandb_tags is not None:
            wandb_kwargs["tags"] = args.wandb_tags

        wandb_run = wandb.init(**wandb_kwargs)
    elif args.wandb_project is not None and wandb is None:
        print("[!] wandb not installed, but --wandb_project was set. "
              "Install wandb or unset --wandb_project to disable logging.")

    try:
        set_global_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[*] Loading model & tokenizer (AutoModel)...")
        model, tokenizer, dcfg = load_model_tokenizer(
            args.tokenizer_config_path,
            args.model_config_path,
            args.dataset_config_path,
            args.ckpt_path,
            device,
        )
        assert dcfg.num_prediction_tasks == 11, f"Expected 11 tasks, got {dcfg.num_prediction_tasks}"

        print("[*] Patching GumbeldoreMixinV2 onto model...")
        patch_gumbeldore_mixin(model)
        print("[*] Overriding _get_model_logits for Hyformer...")
        patch_get_model_logits(model)

        lm_token_id   = tokenizer.task_token_id("lm")
        pred_token_id = tokenizer.task_token_id("prediction")
        bos_id        = tokenizer.bos_token_id
        pad_id        = tokenizer.pad_token_id
        eos_id        = tokenizer.eos_token_id

        if hasattr(tokenizer, "generation_prefix") and tokenizer.generation_prefix is not None:
            prefix_ids = tokenizer.generation_prefix
        else:
            prefix_ids = [lm_token_id, bos_id]

        prefix_input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device)

        oracle = partial(
            peptide_mic_oracle_fn,
            model=model,
            tokenizer=tokenizer,
            pred_token_id=pred_token_id,
            pad_id=pad_id,
            device=device,
            alpha_gneg=args.alpha_gneg,
            beta_gpos=args.beta_gpos,
            baseline_gneg=args.baseline_gneg,
            baseline_gpos=args.baseline_gpos,
            min_len=args.min_len,
            max_len=args.max_len,
        )

        advantage_fn = make_advantage_fn(args.adv_scale)

        print(
            f"[*] Running Gumbeldore: num_rounds={args.num_rounds}, "
            f"beam_width={args.beam_width} (~{args.num_rounds * args.beam_width} samples)"
        )

        with torch.inference_mode():
            raw_samples = model.generate(
                prefix_input_ids=prefix_input_ids,
                oracle_fn=oracle,
                advantage_fn=advantage_fn,
                max_sequence_length=args.max_sequence_length,
                eos_token_id=eos_id,
                advantage_constant=args.advantage_constant,
                beam_width=args.beam_width,
                num_rounds=args.num_rounds,
                min_nucleus_top_p=args.min_nucleus_top_p,
            )

        samples_list: List[torch.Tensor] = []
        if isinstance(raw_samples, torch.Tensor):
            for i in range(raw_samples.size(0)):
                samples_list.append(raw_samples[i].detach().cpu())
        elif isinstance(raw_samples, list):
            if len(raw_samples) == 0:
                print("[!] Empty samples returned.")
                return
            first = raw_samples[0]
            if isinstance(first, torch.Tensor):
                samples_list = [s.detach().cpu() for s in raw_samples]
            else:
                samples_list = [torch.tensor(s, dtype=torch.long) for s in raw_samples]
        else:
            raise TypeError(f"Unexpected type for samples: {type(raw_samples)}")

        num_samples = len(samples_list)
        print(f"[*] Got {num_samples} samples from Gumbeldore.")

        round_to_gneg: Dict[int, List[float]] = {}
        round_to_gpos: Dict[int, List[float]] = {}
        round_to_seqs: Dict[int, List[str]] = {}
        round_to_selectivity: Dict[int, List[float]] = {}
        round_to_lengths: Dict[int, List[int]] = {}

        total_samples = num_samples
        valid_samples = 0
        csv_rows: List[List] = []

        for idx, token_ids in enumerate(samples_list):
            round_idx = idx // args.beam_width

            if token_ids.dim() > 1:
                token_ids = token_ids.view(-1)

            token_ids = token_ids.to(device)
            seq = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True).strip().upper()

            # Only enforce basic charset/length; detailed rules already baked into reward
            if not is_valid_peptide(seq, min_len=args.min_len, max_len=args.max_len):
                continue

            input_ids = token_ids.unsqueeze(0)  # (1, L)
            pred_ids = to_prediction_inputs(input_ids, pred_token_id)
            attn_mask = (pred_ids != pad_id)

            mic_vec = _predict_mic_vec(model, pred_ids, attn_mask)

            mic_all  = float(mic_vec.mean())
            gneg     = float(mic_vec[GNEG_IDXS].mean())
            gpos     = float(mic_vec[GPOS_IDXS].mean())
            min_mic  = float(mic_vec.min())

            if not (np.isfinite(gneg) and np.isfinite(gpos)):
                continue

            selectivity = gneg / gpos if gpos != 0 else gneg
            seq_len = len(seq)

            valid_samples += 1
            round_to_gneg.setdefault(round_idx, []).append(gneg)
            round_to_gpos.setdefault(round_idx, []).append(gpos)
            round_to_seqs.setdefault(round_idx, []).append(seq)
            round_to_selectivity.setdefault(round_idx, []).append(selectivity)
            round_to_lengths.setdefault(round_idx, []).append(seq_len)

            row = (
                [round_idx, idx, seq, seq_len,
                 mic_all, gneg, gpos, selectivity, min_mic] +
                [float(x) for x in mic_vec.tolist()]
            )
            csv_rows.append(row)

        print(f"[*] Total samples: {total_samples}")
        print(f"[*] Valid peptides (charset/length): {valid_samples}")
        if total_samples > 0:
            validity_fraction = valid_samples / total_samples
            print(f"[*] Validity fraction: {validity_fraction:.3f}")
        else:
            validity_fraction = 0.0

        if wandb_run is not None:
            wandb_run.log(
                {
                    "samples/total": total_samples,
                    "samples/valid": valid_samples,
                    "samples/valid_fraction": validity_fraction,
                }
            )

        print("\nPer-round Gram− pMIC stats:")
        for r in sorted(round_to_gneg.keys()):
            vals = np.array(round_to_gneg[r])
            if len(vals) == 0:
                continue
            mean = vals.mean()
            median = np.median(vals)
            std = vals.std()
            vmin = vals.min()
            vmax = vals.max()
            print(
                f"  Round {r}: n={len(vals)}, mean={mean:.3f}, "
                f"median={median:.3f}, std={std:.3f}, "
                f"min={vmin:.3f}, max={vmax:.3f}"
            )
            if wandb_run is not None and wandb is not None:
                wandb_run.log(
                    {
                        "gneg/n": len(vals),
                        "gneg/mean": float(mean),
                        "gneg/median": float(median),
                        "gneg/std": float(std),
                        "gneg/min": float(vmin),
                        "gneg/max": float(vmax),
                        "gneg/hist": wandb.Histogram(vals),
                        "round": r,
                    },
                    step=r,
                )

        print("\nPer-round Gram+ pMIC stats:")
        for r in sorted(round_to_gpos.keys()):
            vals = np.array(round_to_gpos[r])
            if len(vals) == 0:
                continue
            mean = vals.mean()
            median = np.median(vals)
            std = vals.std()
            vmin = vals.min()
            vmax = vals.max()
            print(
                f"  Round {r}: n={len(vals)}, mean={mean:.3f}, "
                f"median={median:.3f}, std={std:.3f}, "
                f"min={vmin:.3f}, max={vmax:.3f}"
            )
            if wandb_run is not None and wandb is not None:
                wandb_run.log(
                    {
                        "gpos/n": len(vals),
                        "gpos/mean": float(mean),
                        "gpos/median": float(median),
                        "gpos/std": float(std),
                        "gpos/min": float(vmin),
                        "gpos/max": float(vmax),
                        "gpos/hist": wandb.Histogram(vals),
                        "round": r,
                    },
                    step=r,
                )

        print("\nPer-round selectivity stats (gneg/gpos):")
        for r in sorted(round_to_selectivity.keys()):
            vals = np.array(round_to_selectivity[r])
            if len(vals) == 0:
                continue
            mean = vals.mean()
            median = np.median(vals)
            std = vals.std()
            vmin = vals.min()
            vmax = vals.max()
            print(
                f"  Round {r}: n={len(vals)}, mean={mean:.3f}, "
                f"median={median:.3f}, std={std:.3f}, "
                f"min={vmin:.3f}, max={vmax:.3f}"
            )
            if wandb_run is not None and wandb is not None:
                wandb_run.log(
                    {
                        "selectivity/n": len(vals),
                        "selectivity/mean": float(mean),
                        "selectivity/median": float(median),
                        "selectivity/std": float(std),
                        "selectivity/min": float(vmin),
                        "selectivity/max": float(vmax),
                        "selectivity/hist": wandb.Histogram(vals),
                        "round": r,
                    },
                    step=r,
                )

        print("\nPer-round sequence length stats:")
        for r in sorted(round_to_lengths.keys()):
            vals = np.array(round_to_lengths[r])
            if len(vals) == 0:
                continue
            mean = vals.mean()
            median = np.median(vals)
            std = vals.std()
            vmin = vals.min()
            vmax = vals.max()
            print(
                f"  Round {r}: n={len(vals)}, mean_len={mean:.3f}, "
                f"median_len={median:.3f}, std_len={std:.3f}, "
                f"min_len={vmin:.3f}, max_len={vmax:.3f}"
            )
            if wandb_run is not None and wandb is not None:
                wandb_run.log(
                    {
                        "length/n": len(vals),
                        "length/mean": float(mean),
                        "length/median": float(median),
                        "length/std": float(std),
                        "length/min": float(vmin),
                        "length/max": float(vmax),
                        "length/hist": wandb.Histogram(vals),
                        "round": r,
                    },
                    step=r,
                )

        report_diversity_per_round(
            round_to_seqs,
            max_pairs_per_round=args.max_div_pairs_per_round,
            out_csv=args.out_diversity_csv,
            wandb_run=wandb_run,
        )

        diversity_plot_path = None
        if args.out_plot is not None:
            base, ext = os.path.splitext(args.out_plot)
            diversity_plot_path = base + "_diversity" + ext
            plot_diversity_by_round(
                round_to_seqs,
                max_pairs_per_round=args.max_div_pairs_per_round,
                out_png=diversity_plot_path,
                title="Per-round diversity (Levenshtein)",
            )

        if csv_rows and args.out_csv:
            out_dir = os.path.dirname(args.out_csv)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            header = (
                ["round", "global_idx", "sequence", "length",
                 "mean_all_pMIC", "mean_gneg_pMIC", "mean_gpos_pMIC",
                 "selectivity_score", "min_pMIC"]
                + [f"{lab}_pMIC" for lab in LABELS]
            )

            with open(args.out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(csv_rows)

            print(f"[+] Wrote CSV to: {args.out_csv}")

        plot_gneg_gpos_by_round(round_to_gneg, round_to_gpos, out_png=args.out_plot)

        if wandb_run is not None:
            if args.out_csv and os.path.exists(args.out_csv):
                seq_art = wandb.Artifact("gumbeldore_sequences", type="dataset")
                seq_art.add_file(args.out_csv)
                wandb_run.log_artifact(seq_art)

            if args.out_diversity_csv and os.path.exists(args.out_diversity_csv):
                div_art = wandb.Artifact("gumbeldore_diversity", type="dataset")
                div_art.add_file(args.out_diversity_csv)
                wandb_run.log_artifact(div_art)

            if args.out_plot and os.path.exists(args.out_plot):
                wandb_run.log({"plots/gram_boxplot": wandb.Image(args.out_plot)})

            if diversity_plot_path is not None and os.path.exists(diversity_plot_path):
                wandb_run.log(
                    {"plots/diversity_per_round": wandb.Image(diversity_plot_path)}
                )

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()