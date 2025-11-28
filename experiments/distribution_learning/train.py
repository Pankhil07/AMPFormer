""" Run the training script for the distribution learning task. 

Handles both single-GPU and DDP training.
"""

import os, logging, argparse, sys
from typing import Dict, Optional, Union, List, Tuple, Any
import torch
import torch.distributed as dist
from sklearn.metrics import r2_score, mean_squared_error               # NEW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from scipy.stats import pearsonr, spearmanr                            # NEW
import numpy as np      
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.trainer import TrainerConfig
from hyformer.configs.logger import LoggerConfig

from hyformer.utils.datasets.auto import AutoDataset
from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.models.auto import AutoModel
from hyformer.utils.loggers.auto import AutoLogger

from hyformer.trainers.trainer import Trainer

from hyformer.utils.experiments import log_args, dump_configs
from hyformer.utils.reproducibility import set_seed

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
# logging.captureWarnings(False)

torch.set_float32_matmul_precision('high')
def regression_stats(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     sentinel: float = -1.0) -> Dict[str, float]:
    """Return RMSE, R², Pearson, Spearman for valid entries only."""
    mask  = y_true != sentinel
    if mask.sum() == 0:
        return {k: float("nan") for k in ("rmse", "r2", "pearson", "spearman")}

    yt, yp = y_true[mask], y_pred[mask]

    rmse     = np.sqrt(mean_squared_error(yt, yp))
    r2       = r2_score(yt, yp)
    pearson  = pearsonr(yt, yp).statistic if mask.sum() >= 2 else float("nan")
    spearman = spearmanr(yt, yp).statistic if mask.sum() >= 2 else float("nan")
    return {"rmse": rmse, "r2": r2, "pearson": pearson, "spearman": spearman}

def decode_and_metrics(y_true, logits):
    """
    Works for binary, multiclass, multilabel, multi‑task‑multiclass.
    Returns metrics dict + decoded labels with shape matching y_true.
    """
    y_true = np.asarray(y_true)
    logits = np.asarray(logits)

    # ---------- decide task ----------
    if y_true.ndim == 1:                          # binary or single‑label multiclass
        binary = (np.unique(y_true) <= 1).all()
        if binary:
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
            y_pred = (probs >= 0.5).astype(int)
            auc = roc_auc_score(y_true, probs)
        else:                                     # multiclass single label
            probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)
            auc = roc_auc_score(y_true, probs, multi_class="ovr")
    else:                                         # y_true is 2‑D
        multilabel = set(np.unique(y_true)) <= {0, 1}
        if multilabel:
            probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
            y_pred = (probs >= 0.5).astype(int)
            auc = roc_auc_score(y_true, probs, average="macro")
        else:                                     # multi‑task multiclass
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            y_pred = np.argmax(probs, axis=-1)
            # AUC rarely defined here – skip or compute per task
            auc = float("nan")

    # ---------- general metrics ----------
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true.reshape(-1), y_pred.reshape(-1),
        average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r,
            "f1": f1, "auc": auc}, y_pred
def classification_stats(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         average: str = 'macro',
                         sentinel: float = -1.0) -> Dict[str, float]:
    """
    y_true: integer labels, shape (n_samples,) or (n_samples, n_labels)
    y_pred: either integer predictions or float scores for AUC, same shape
    """
    # mask out sentinel if necessary
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    mask = y_true != sentinel
    yt = y_true[mask]
    yp = y_pred[mask]

    # If you have probabilities/scores for AUC,
    # you might compute roc_auc_score per label or averaged.
    stats = {}
    stats['accuracy'] = accuracy_score(yt, yp)

    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average=average, zero_division=0
    )
    stats.update({'precision': p, 'recall': r, 'f1': f1})
    # optional: if yp are scores, compute AUC
    try:
        stats['auc'] = roc_auc_score(yt, yp, average=average)
    except Exception:
        stats['auc'] = float('nan')
    return stats


def per_label_stats(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    sentinel: float = -1.0) -> Tuple[List[Dict[str, float]],
                                                      Dict[str, float]]:
    """
    Returns
      - a list of metric-dicts, one per label (shape = (n_labels,))
      - a dict with macro-averaged metrics
    """
    if y_true.ndim == 1:          # single-label fallback
        stats = regression_stats(y_true, y_pred, sentinel)
        return [stats], stats

    label_stats = [
        regression_stats(y_true[:, j], y_pred[:, j], sentinel)
        for j in range(y_true.shape[1])
    ]

    # macro-average (ignore nans)
    avg_stats = {
        k: float(np.nanmean([ls[k] for ls in label_stats]))
        for k in label_stats[0].keys()
    }
    return label_stats, avg_stats


@record
def main(args):    

    # Create output directory
    if args.out_dir is not None and not os.path.exists(args.out_dir) and int(os.environ.get('LOCAL_RANK', 0)) == 0:
        if args.debug:
            os.path.join(args.out_dir, "debug")
        os.makedirs(args.out_dir, exist_ok=False)

    # Load configurations
    dataset_config = DatasetConfig.from_config_filepath(args.dataset_config_path)
    tokenizer_config = TokenizerConfig.from_config_filepath(args.tokenizer_config_path)
    model_config = ModelConfig.from_config_filepath(args.model_config_path)
    trainer_config = TrainerConfig.from_config_filepath(args.trainer_config_path)
    logger_config = LoggerConfig.from_config_filepath(args.logger_config_path) if args.logger_config_path else None
    
    # Set debug mode
    if args.debug:
        model_config.num_transformer_layers = 2
        trainer_config.max_epochs = 2
        trainer_config.log_interval = 2
        trainer_config.warmup_iters = 200
    
    # Set learning rate
    if args.learning_rate is not None:
        trainer_config.learning_rate = args.learning_rate
        console.info(f"Learning rate set to: {args.learning_rate}")
    
    # Store configs within the output directory, for reproducibility
    #if args.out_dir is not None:
    #    map(lambda config_file: config_file.save(os.path.join(args.out_dir, f'{config_file.__class__.__name__}.json')), [dataset_config, tokenizer_config, model_config, trainer_config])

    if args.out_dir is not None:
        for cfg in (dataset_config, tokenizer_config, model_config, trainer_config):
            cfg.save(os.path.join(args.out_dir, f'{cfg.__class__.__name__}.json'))
    # Initialize
    train_dataset = AutoDataset.from_config(dataset_config, split='train', root=args.data_dir)
    val_dataset = AutoDataset.from_config(dataset_config, split='val', root=args.data_dir)
    test_dataset = AutoDataset.from_config(dataset_config, split='test', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model = AutoModel.from_config(
        model_config, prediction_task_type=dataset_config.prediction_task_type, num_prediction_tasks=dataset_config.num_prediction_tasks
        )
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    
    # Check for tokenizer and model vocabulary size mismatch
    assert len(tokenizer) == model.vocab_size, f"Tokenizer vocab size {len(tokenizer)} does not match model vocab size {model.vocab_size}"
    
    # Set debug mode
    if args.debug:
        train_dataset.data, train_dataset.target = train_dataset.data[:1500], train_dataset.target[:1500] if train_dataset.target is not None else None
        val_dataset.data, val_dataset.target = val_dataset.data[:1500], val_dataset.target[:1500] if val_dataset.target is not None else None
    
    # Store configs within the logger object, for reproducibility
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config)

    # Determine the device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    # Determine whether to use DDP
    if trainer_config.enable_ddp and int(os.environ.get('RANK', -1)) != -1 and torch.cuda.device_count() > 1:
        print("Running in distributed setting...", flush=True)
        init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f'cuda:{int(os.environ["LOCAL_RANK"])}')
        seed = args.experiment_seed + int(os.environ['LOCAL_RANK'])
        print(f"Rank: {int(os.environ['RANK'])}, Local Rank: {int(os.environ['LOCAL_RANK'])}, Device: {device}", flush=True)

    if logger is not None:
        logger.init_run()
        logger.watch_model(model)

    # Initialize trainer
    trainer = Trainer(
        config=trainer_config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        out_dir=args.out_dir,
        logger=logger,
        worker_seed=args.experiment_seed
        )

    if args.model_ckpt_path:
        trainer.resume_from_checkpoint(args.model_ckpt_path, resume_training=args.resume_training,discard_prediction_head = True)
        console.info(f"Resuming pre-trained model from {args.model_ckpt_path}")
    else:
        console.info("Training from scratch")

    # Load pre-trained model if specified
    #if args.model_ckpt_path:
    #    console.info(f"Loading backbone weights, discarding old head, from {args.model_ckpt_path}")
    #    model.load_pretrained(
    #        filepath = args.model_ckpt_path,
     #       device = device,
     #       discard_prediction_head=True,
     #   )

    # Ensure all processes are ready before training
    if trainer._is_distributed_run:
        dist.barrier() 

    # Run training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_specific_validation=args.task_specific_validation,
        patience=args.patience,
        )

    if test_dataset is not None and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        # 1) reload best weights
        best_ckpt = os.path.join(args.out_dir, "ckpt.pt")
        trainer.resume_from_checkpoint(best_ckpt,
                                    resume_training=False,
                                    discard_prediction_head=False)

        # 2) run the forward pass once
        test_loader = trainer.create_loader(
            test_dataset,
            tasks={'prediction': 1.0},
            shuffle=False
        )

        y_true, y_pred = [], []
        with torch.inference_mode():
            for batch in test_loader:
                targets = batch['target']
                batch   = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                logits  = trainer.model(**batch)['logits'].cpu()

                y_true.append(targets.detach().cpu().numpy())
                y_pred.append(logits.detach().cpu().numpy())
                

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        # 3) undo any scaler attached to the *train* dataset
        scaler = getattr(train_dataset, "target_transform", None)
        if scaler is not None:
            y_true = scaler.inverse_transform(y_true)
            y_pred = scaler.inverse_transform(y_pred)

        # after concatenating y_true, y_pred and undoing any scaler…
        task = dataset_config.prediction_task_type
        if task == 'regression':
            label_stats, avg_stats = per_label_stats(y_true, y_pred, sentinel=-1.0)
            # per‐label
            for idx, st in enumerate(label_stats):
                console.info(
                    f"[FINAL TEST] Label {idx:02d}: RMSE={st['rmse']:.4f}, "
                    f"R²={st['r2']:.4f}, Pearson={st['pearson']:.4f}, Spearman={st['spearman']:.4f}"
                )
                logger.log({
                    f"test/label{idx:02d}/rmse":      st["rmse"],
                    f"test/label{idx:02d}/r2":       st["r2"],
                    f"test/label{idx:02d}/pearson":  st["pearson"],
                    f"test/label{idx:02d}/spearman": st["spearman"],
                })
            # macro
            console.info(
                f"[TEST AVG] R²={avg_stats['r2']:.4f}, Pearson={avg_stats['pearson']:.4f}, "
                f"Spearman={avg_stats['spearman']:.4f}, RMSE={avg_stats['rmse']:.4f}"
            )
            logger.log({
                "test/avg/rmse":      avg_stats["rmse"],
                "test/avg/r2":       avg_stats["r2"],
                "test/avg/pearson":  avg_stats["pearson"],
                "test/avg/spearman": avg_stats["spearman"],
            })
            if logger is not None:
                logger.finish()
        elif task == 'classification':

            metrics, y_pred_labels = decode_and_metrics(y_true, y_pred)
            console.info(
                f"[FINAL TEST] Acc={metrics['accuracy']:.4f}, "
                f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}"
            )
            logger.log({f"test/{k}": v for k, v in metrics.items()})

            # Flatten a (N,1) logits array to (N,)
            #if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            #    raw_logits = y_pred.squeeze(1)
            #else:
            #    raw_logits = y_pred

            # Compute true probabilities via sigmoid
            #probs = torch.sigmoid(torch.from_numpy(raw_logits)).numpy()
            # Threshold at 0.5 to get discrete predictions
            #y_pred_labels = (probs >= 0.5).astype(int)

            # Now compute each metric explicitly
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

            #acc = accuracy_score(y_true, y_pred_labels)
            #p, r, f1, _ = precision_recall_fscore_support(
            #    y_true, y_pred_labels, average='binary', zero_division=0
            #)
            #auc = roc_auc_score(y_true, probs)

            #console.info(
            #    f"[TEST CLASS] Acc={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
            #)
            #logger.log({
            #    "test/accuracy": acc,
            #    "test/precision": p,
            #    "test/recall": r,
            #    "test/f1": f1,
            #    "test/auc": auc,
            #})

            if logger is not None:
                logger.finish()
    # Clean up
    if trainer_config.enable_ddp and int(os.environ.get('RANK', -1)) != -1:
        destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--experiment_seed", type=int, default=0, help="Seed for the experiment")
    parser.add_argument("--dataset_config_path", type=str, required=True, help="Path to the dataset config file")
    parser.add_argument("--tokenizer_config_path", type=str, required=True, help="Path to the tokenizer config file")
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--trainer_config_path", type=str, required=True, help="Path to the trainer config file")
    parser.add_argument("--logger_config_path", type=str, nargs='?', help="Path to the logger config file")
    parser.add_argument("--model_ckpt_path", type=str, nargs='?', help="Path to the model checkpoint file")
    parser.add_argument("--resume_training", default=False, action=argparse.BooleanOptionalAction, help="Resume training from the checkpoint file")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction, help="Run in debug mode")
    parser.add_argument("--task_specific_validation", type=str, nargs='?', help="Task with respect to which validation is performed")
    parser.add_argument("--patience", type=int, nargs='?', help="Number of epochs to wait before early stopping")
    parser.add_argument("--use_deterministic_algorithms", default=False, action=argparse.BooleanOptionalAction, help="Use deterministic algorithms")
    parser.add_argument("--learning_rate", type=float, nargs='?', help="Learning rate")
    args = parser.parse_args()
    log_args(args)
    return args

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    args = parse_args()
    set_seed(args.experiment_seed, use_deterministic_algorithms=args.use_deterministic_algorithms)
    main(args)
    