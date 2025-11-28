import os
import csv
import gzip
import hashlib
import random
import argparse
from typing import List
import numpy as np
import torch
import torch.nn as nn  

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.dataset import DatasetConfig
from hyformer.models.auto import AutoModel
from hyformer.utils.tokenizers.auto import AutoTokenizer

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

# --------------------------- helpers ---------------------------
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def is_valid_peptide(seq: str, min_len=2, max_len=50) -> bool:
    if not seq or not (min_len <= len(seq) <= max_len):
        return False
    return all(ch in "ACDEFGHIKLMNPQRSTVWY" for ch in seq)

def write_header_csv_gz(path: str, header: List[str]):
    """Write CSV header once, safely, even if multiple procs try at once."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    tmp = f"{path}.tmp.{os.getpid()}_{random.randint(0, 1_000_000)}"
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

    try:
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def append_rows_csv_gz(path: str, rows: List[List]):
    with gzip.open(path, "at", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

def load_model_tokenizer(
    tokenizer_config_path: str,
    model_config_path: str,
    dataset_config_path: str,
    ckpt_path: str,
    device: torch.device,
):
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
    return model, tokenizer, dset_cfg

def to_prediction_inputs(gen_ids: torch.Tensor, pred_task_token_id: int) -> torch.Tensor:
    pred_ids = gen_ids.clone()
    pred_ids[:, 0] = pred_task_token_id
    return pred_ids

@torch.inference_mode()
def predict_no_dropout(model, input_ids, attention_mask):
    """
    Deterministic, single forward pass (dropout disabled).
    Returns mean and std arrays with shape (B, 11). std is zeros.
    """
    was_training = model.training
    model.eval()  # disable dropout
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task="prediction",
        return_loss=False,
    )
    mean = out["logits"].detach().cpu().numpy()                 # (B, 11)
    std  = np.zeros_like(mean, dtype=np.float32)                # keep CSV layout
    model.train(was_training)
    return mean, std

def set_global_determinism(enable: bool):
    if not enable:
        return
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False

# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_config_path", required=True)
    ap.add_argument("--model_config_path", required=True)
    ap.add_argument("--dataset_config_path", required=True)
    ap.add_argument("--ckpt_path", required=True, help="Path to ckpt.pt (best model)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--sequences", type=int, default=1_000_000, help="rows to write in THIS process")
    ap.add_argument("--batch_size", type=int, default=1024)

    ap.add_argument("--gen_len", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--dedup_in_shard", action="store_true")

    ap.add_argument("--min_value", type=float, default=None, help="Clamp lower bound for predictions")
    ap.add_argument("--max_value", type=float, default=None, help="Clamp upper bound for predictions")

    ap.add_argument("--deterministic", action="store_true", help="Enable PyTorch deterministic algorithms")

    args = ap.parse_args()

    set_global_determinism(args.deterministic)

    # --- unique shard id per process across arrays and tasks ---
    array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    proc_id  = int(os.environ.get("SLURM_PROCID", "0"))        # rank within the srun
    ntasks   = int(os.environ.get("SLURM_NTASKS", "1"))        # tasks per array index
    shard_id = array_id * ntasks + proc_id                     # global, collision-free

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # per-shard deterministic seed (so shards differ)
    base_seed = 49 + 9949 * shard_id
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    model, tokenizer, dcfg = load_model_tokenizer(
        args.tokenizer_config_path,
        args.model_config_path,
        args.dataset_config_path,
        args.ckpt_path,
        device,
    )
    assert dcfg.num_prediction_tasks == 11, f"Expected 11 tasks, got {dcfg.num_prediction_tasks}"

    lm_token_id   = tokenizer.task_token_id("lm")
    pred_token_id = tokenizer.task_token_id("prediction")
    bos_id        = tokenizer.bos_token_id
    pad_id        = tokenizer.pad_token_id
    eos_id        = tokenizer.eos_token_id

    prefix = torch.tensor([[lm_token_id, bos_id]], dtype=torch.long, device=device)

    # Prepare output 
    out_path = os.path.join(args.out_dir, f"gen_preds_shard-{shard_id:06d}.csv.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[shard {shard_id}] exists, skipping: {out_path}")
        return

    # header: aggregates + per-label MEAN + per-label STD + uncertainty aggregates + meta
    header = (
        ["sequence", "length",
         "mean_all", "mean_gneg", "mean_gpos", "min_mic"] +
        [f"{lab} (mean)" for lab in LABELS] +
        [f"{lab} (std)"  for lab in LABELS] +
        ["std_all_mean", "std_gneg_mean", "std_gpos_mean", "std_max",
         "shard", "seed", "sha1"]
    )
    write_header_csv_gz(out_path, header)

    seen = set() if args.dedup_in_shard else None
    written = 0

    while written < args.sequences:
        need = min(args.batch_size, args.sequences - written)
        batched_prefix = prefix.repeat(need, 1)

        # ---- GENERATION ----
        model.eval()
        with torch.inference_mode():
            gen_ids = model.generate(
                prefix_input_ids=batched_prefix,
                num_tokens_to_generate=args.gen_len,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=False,
            )

        pred_ids  = to_prediction_inputs(gen_ids, pred_token_id)
        attn_mask = (pred_ids != pad_id).long()

        mean_pred, std_pred = predict_no_dropout(
            model=model,
            input_ids=pred_ids.to(device),
            attention_mask=attn_mask.to(device),
        )  # each (B, 11), std is zeros

        seqs = [tokenizer.decode(x.tolist(), skip_special_tokens=True).strip().upper()
                for x in gen_ids]

        rows = []
        for s, p_mean, p_std in zip(seqs, mean_pred, std_pred):
            if not is_valid_peptide(s, args.min_len, args.max_len):
                continue
            h = sha1(s)
            if seen is not None and h in seen:
                continue
            if seen is not None:
                seen.add(h)

            length   = len(s)
            p_mean   = np.asarray(p_mean, dtype=np.float32)  # (11,)
            p_std    = np.asarray(p_std,  dtype=np.float32)  # (11,) zeros

            # Optional clamping to avoid negatives etc.
            #if args.min_value is not None:
            #    p_mean = np.maximum(p_mean, args.min_value)
            #if args.max_value is not None:
            #    p_mean = np.minimum(p_mean, args.max_value)

            mean_all  = float(p_mean.mean())
            mean_gneg = float(p_mean[GNEG_IDXS].mean())
            mean_gpos = float(p_mean[GPOS_IDXS].mean())
            min_mic   = float(p_mean.min())

            std_all_mean  = float(p_std.mean())
            std_gneg_mean = float(p_std[GNEG_IDXS].mean())
            std_gpos_mean = float(p_std[GPOS_IDXS].mean())
            std_max       = float(p_std.max())

            row = (
                [s, length, mean_all, mean_gneg, mean_gpos, min_mic] +
                [float(x) for x in p_mean.tolist()] +
                [float(x) for x in p_std.tolist()] +
                [std_all_mean, std_gneg_mean, std_gpos_mean, std_max,
                 int(shard_id), int(base_seed), h]
            )
            rows.append(row)

        if rows:
            append_rows_csv_gz(out_path, rows)
            written += len(rows)
            print(f"[shard {shard_id}] {written}/{args.sequences}")

        del gen_ids, pred_ids, attn_mask, mean_pred, std_pred
        torch.cuda.empty_cache()

    print(f"[shard {shard_id}] DONE → {out_path}")

if __name__ == "__main__":
    main()
