"""
scripts/tune_optuna.py
----------------------
Hyperparameter Tuning for HMST on UWIQA (or SAUD) using Optuna.

Features:
- Search space: Modern optimizers (Lion, SAM-AdamW, AdamW, Adam, SGD),
  learning rate, weight decay, batch size, loss weights, ranking margin,
  dropout, transformer depth/width, and augmentations.
- Bayesian Optimization via TPESampler.
- Aggressive pruning via MedianPruner to terminate unpromising trials early.
- SQLite study persistence (resumeable, pauseable, inspectable via optuna-dashboard).
- Automatically exports the winning parameters to configs/uwiqa_tuned.yaml.

Usage:
    # Run 30 trials on UWIQA
    python scripts/tune_optuna.py --trials 30 --epochs 25

    # Run with custom storage
    python scripts/tune_optuna.py --trials 50 --storage sqlite:///outputs/uwiqa/optuna/uwiqa_study.db

    # Launch dashboard in browser
    optuna-dashboard sqlite:///outputs/uwiqa/optuna/hmst_uwiqa_tuning.db
"""

import os
import sys
import argparse
import random
import time
import json
import yaml
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

try:
    # pyrefly: ignore[missing-import]
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("\n[Error] Optuna is not installed in your Python environment.")
    print("Please install it by running:")
    print("    pip install optuna optuna-dashboard")
    sys.exit(1)

from data import (
    UWIQADataset,
    SAUDDataset,
    UIDDataset,
    make_uwiqa_splits,
    make_saud_splits,
    make_uid_splits,
    get_train_transforms,
    get_val_transforms,
)
from models import HMST, HMSTLoss, build_correspondence_matrix
from training import compute_metrics, build_optimizer, SAM

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def make_absolute(path: str, base: str) -> str:
    if os.path.isabs(path):
        resolved = path
    else:
        resolved = os.path.normpath(os.path.join(base, path))
    if os.path.exists(resolved):
        return resolved
    # Fallback for parent dir variations
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-2] == "UWIQA" and "UWIQA dataset" in parts[-1]:
        alt = path.replace("UWIQA/UWIQA dataset", "UWIQA dataset")
        alt_res = os.path.normpath(os.path.join(base, alt)) if not os.path.isabs(alt) else alt
        if os.path.exists(alt_res):
            return alt_res
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Objective
# ─────────────────────────────────────────────────────────────────────────────

class Objective:
    def __init__(self, args, project_root: str):
        self.args = args
        self.project_root = project_root
        self.device = get_device(args.device)

        # Load base configuration template
        config_path = os.path.join(project_root, "configs", "full.yaml")
        with open(config_path, "r") as f:
            self.base_cfg = yaml.safe_load(f)

        self.dataset_name = args.dataset or self.base_cfg.get("dataset", "uwiqa")

        # Resolve dataset paths
        if self.dataset_name == "uwiqa":
            cfg = self.base_cfg["uwiqa"]
            self.image_dir = make_absolute(cfg["image_dir"], project_root)
            self.score_file = make_absolute(cfg["score_file"], project_root)
            self.train_idx, self.test_idx = make_uwiqa_splits(
                self.image_dir, self.score_file, seed=args.seed
            )
            val_size = max(1, len(self.train_idx) // 5)
            self.val_idx = self.train_idx[-val_size:]
            self.train_idx = self.train_idx[:-val_size]

        elif self.dataset_name == "saud":
            cfg = self.base_cfg["saud"]
            self.benchmark_dir = make_absolute(cfg["benchmark_dir"], project_root)
            self.bt_mat = make_absolute(cfg["bt_scores_mat"], project_root)
            self.train_idx, self.test_idx = make_saud_splits(
                self.bt_mat, seed=args.seed
            )
            val_size = max(1, len(self.train_idx) // 5)
            self.val_idx = self.train_idx[-val_size:]
            self.train_idx = self.train_idx[:-val_size]

        elif self.dataset_name == "uid":
            cfg = self.base_cfg["uid"]
            self.image_dir  = make_absolute(cfg["image_dir"],  project_root)
            self.score_file = make_absolute(cfg["score_file"], project_root)
            self.train_idx, self.test_idx = make_uid_splits(
                self.image_dir, self.score_file, seed=args.seed
            )
            val_size  = max(16, (len(self.train_idx) // 5) // 16 * 16)   # whole scenes of 16 images
            self.val_idx   = self.train_idx[-val_size:]
            self.train_idx = self.train_idx[:-val_size]

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __call__(self, trial: optuna.Trial) -> float:
        # Seed this trial run
        set_seed(self.args.seed)

        # ── 1. Suggest Hyperparameters ─────────────────────────────────────────
        # Optimizer selection
        optimizer_type = trial.suggest_categorical(
            "optimizer", ["lion", "adamw", "adam", "sgd", "sam_adamw"]
        )

        # Conditional learning rate and weight decay per optimizer
        if optimizer_type == "lion":
            lr = trial.suggest_float("lr_lion", 5.0e-6, 5.0e-5, log=True)
            weight_decay = trial.suggest_float("wd_lion", 1.0e-2, 1.0e-1, log=True)
            opt_kwargs = {}
        elif optimizer_type == "adamw":
            lr = trial.suggest_float("lr_adamw", 3.0e-5, 3.0e-4, log=True)
            weight_decay = trial.suggest_float("wd_adamw", 1.0e-4, 5.0e-3, log=True)
            opt_kwargs = {}
        elif optimizer_type == "adam":
            lr = trial.suggest_float("lr_adam", 3.0e-5, 3.0e-4, log=True)
            weight_decay = trial.suggest_float("wd_adam", 1.0e-6, 1.0e-3, log=True)
            opt_kwargs = {}
        elif optimizer_type == "sgd":
            lr = trial.suggest_float("lr_sgd", 1.0e-3, 1.0e-2, log=True)
            weight_decay = trial.suggest_float("wd_sgd", 1.0e-4, 1.0e-3, log=True)
            opt_kwargs = {"momentum": 0.9, "nesterov": True}
        elif optimizer_type == "sam_adamw":
            lr = trial.suggest_float("lr_sam", 3.0e-5, 2.0e-4, log=True)
            weight_decay = trial.suggest_float("wd_sam", 1.0e-4, 5.0e-3, log=True)
            rho = trial.suggest_float("rho_sam", 0.01, 0.05)
            opt_kwargs = {"rho": rho}

        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        scheduler_name = trial.suggest_categorical("scheduler", ["none", "cosine", "step"])

        # Loss weights
        lambda_mos = 1.0
        lambda_rank = trial.suggest_float("lambda_rank", 0.5, 2.0, step=0.25)
        lambda_cons = trial.suggest_float("lambda_cons", 0.01, 0.3, log=True)
        ranking_margin = trial.suggest_float("ranking_margin", 0.05, 0.20, step=0.05)

        # Architecture & Regularization
        dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
        d_model = trial.suggest_categorical("d_model", [256, 512])
        nhead = 8 if d_model == 512 else 4
        nlayers = trial.suggest_int("nlayers", 1, 3)

        # Augmentation
        rotation_deg = trial.suggest_int("rotation_deg", 5, 15, step=5)
        vflip_prob = trial.suggest_float("vflip_prob", 0.0, 0.5, step=0.1)

        # ── 2. Build Datasets & Loaders ────────────────────────────────────────
        image_size = 224
        train_tf = get_train_transforms(
            image_size=image_size,
            hflip_p=0.5,
            vflip_p=vflip_prob,
            degrees=rotation_deg,
        )
        val_tf = get_val_transforms(image_size=image_size)

        if self.dataset_name == "uwiqa":
            train_ds = UWIQADataset(self.image_dir, self.score_file, self.train_idx, train_tf)
            val_ds   = UWIQADataset(self.image_dir, self.score_file, self.val_idx,   val_tf)
        elif self.dataset_name == "uid":
            train_ds = UIDDataset(self.image_dir, self.score_file, self.train_idx, train_tf)
            val_ds   = UIDDataset(self.image_dir, self.score_file, self.val_idx,   val_tf)
        else:  # saud
            train_ds = SAUDDataset(self.benchmark_dir, self.bt_mat, self.train_idx, train_tf)
            val_ds   = SAUDDataset(self.benchmark_dir, self.bt_mat, self.val_idx,   val_tf)

        pin_mem = self.device.type == "cuda"
        num_workers = self.args.workers

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_mem,
            drop_last=(len(train_ds) > batch_size),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem,
        )

        # ── 3. Build Model & Loss ──────────────────────────────────────────────
        patch_sizes = [32, 64, 128]
        model = HMST(
            image_size=image_size,
            patch_sizes=patch_sizes,
            d_model=d_model,
            nhead=nhead,
            nlayers=nlayers,
            n_regions=4,
            dropout=dropout,
            print_shapes=False,
        ).to(self.device)

        M = build_correspondence_matrix(
            num_patches_fine=model.num_patches[0],
            num_patches_coarse=model.num_patches[2],
            grid_fine=model.grid_sizes[0],
            grid_coarse=model.grid_sizes[2],
        ).to(self.device)

        criterion = HMSTLoss(
            M=M,
            lambda_mos=lambda_mos,
            lambda_rank=lambda_rank,
            lambda_cons=lambda_cons,
            margin=ranking_margin,
        )

        # ── 4. Build Optimizer & Scheduler ─────────────────────────────────────
        optimizer = build_optimizer(
            model,
            optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            **opt_kwargs,
        )

        is_sam = isinstance(optimizer, SAM)
        base_opt_for_sched = optimizer.base_optimizer if is_sam else optimizer

        scheduler = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                base_opt_for_sched, T_max=self.args.epochs, eta_min=lr * 0.01
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                base_opt_for_sched, step_size=max(5, self.args.epochs // 3), gamma=0.5
            )

        use_amp = self.device.type == "cuda" and not is_sam
        scaler = GradScaler("cuda") if use_amp else None

        # ── 5. Trial Training Loop ─────────────────────────────────────────────
        best_val_srocc = -1.0
        patience_counter = 0
        patience = self.args.patience

        try:
            for epoch in range(self.args.epochs):
                model.train()

                for batch in train_loader:
                    images = batch["image"].to(self.device)
                    scores = batch["score"].to(self.device)

                    if is_sam:
                        optimizer.zero_grad()
                        # SAM requires 2 forward/backward passes
                        output = model(images)
                        loss_dict = criterion(output, scores)
                        loss_dict["loss"].backward()
                        optimizer.first_step(zero_grad=True)

                        output2 = model(images)
                        loss_dict2 = criterion(output2, scores)
                        loss_dict2["loss"].backward()
                        optimizer.second_step(zero_grad=True)

                    elif use_amp:
                        optimizer.zero_grad()
                        with autocast("cuda"):
                            output = model(images)
                            loss_dict = criterion(output, scores)
                        scaler.scale(loss_dict["loss"]).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    else:
                        optimizer.zero_grad()
                        output = model(images)
                        loss_dict = criterion(output, scores)
                        loss_dict["loss"].backward()
                        optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # ── Validation ─────────────────────────────────────────────────
                model.eval()
                preds_list, tgts_list = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch["image"].to(self.device)
                        scrs = batch["score"].to(self.device)
                        out = model(imgs)
                        preds_list.append(out["Q_hat"].cpu().numpy())
                        tgts_list.append(scrs.cpu().numpy())

                preds = np.concatenate(preds_list)
                tgts = np.concatenate(tgts_list)
                metrics = compute_metrics(preds, tgts)
                val_srocc = float(metrics["srocc"])
                val_plcc = float(metrics["plcc"])

                # Primary target metric
                trial_metric = val_srocc
                if trial_metric > best_val_srocc:
                    best_val_srocc = trial_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                # ── Optuna Reporting & Pruning ─────────────────────────────────
                trial.report(val_srocc, step=epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping inside trial
                if patience_counter >= patience and epoch >= 10:
                    break

        finally:
            # Prevent CUDA memory accumulation across trials
            del model, optimizer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return best_val_srocc


# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HMST Optuna Hyperparameter Tuning")
    parser.add_argument("--trials", type=int, default=30, help="Number of trials to run")
    parser.add_argument("--epochs", type=int, default=25, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience per trial")
    parser.add_argument("--dataset", type=str, default="uwiqa", choices=["uwiqa", "saud", "uid"])
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split consistency")
    parser.add_argument(
        "--study-name", type=str, default=None,
        help="Name of the Optuna study (default: hmst_<dataset>_tuning)"
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Storage URL (default: sqlite:///outputs/<dataset>/optuna/<study>.db)"
    )
    parser.add_argument(
        "--output-config", type=str, default=None,
        help="Path where the best config YAML will be written (default: configs/<dataset>_tuned.yaml)"
    )
    args = parser.parse_args()

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    dataset = args.dataset
    if args.study_name is None:
        args.study_name = f"hmst_{dataset}_tuning"
    if args.output_config is None:
        args.output_config = f"configs/{dataset}_tuned.yaml"

    # Dataset-scoped Optuna artifacts: outputs/{dataset}/optuna/
    optuna_dir = os.path.join(project_root, "outputs", dataset, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)

    # Default SQLite storage for resumeability
    storage_url = args.storage
    if storage_url is None:
        db_path = os.path.join(optuna_dir, f"{args.study_name}.db").replace("\\", "/")
        storage_url = f"sqlite:///{db_path}"

    print(f"\n{'='*70}")
    print(f"HMST Hyperparameter Tuning with Optuna")
    print(f"{'='*70}")
    print(f"Dataset:       {args.dataset.upper()}")
    print(f"Total Trials:  {args.trials}")
    print(f"Epochs/Trial:  {args.epochs} (patience={args.patience})")
    print(f"Study Name:    {args.study_name}")
    print(f"Storage:       {storage_url}")
    print(f"Output Config: {args.output_config}")
    print(f"{'='*70}\n")

    # Sampler & Pruner
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=8, interval_steps=1)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = Objective(args, project_root)

    start_time = time.time()
    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[Tuning] Process interrupted by user. Saving best results so far...")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TUNING FINISHED in {total_time/60:.1f} minutes")
    print(f"{'='*70}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("\n[Tuning] No trials were successfully completed yet.")
        print("Run the command again to continue sampling trials into the study database.")
        return

    best_trial = study.best_trial
    print(f"\n★ Best Trial #{best_trial.number}")
    print(f"  Validation SROCC: {best_trial.value:.4f}")
    print(f"\nBest Parameters:")
    for k, v in best_trial.params.items():
        print(f"  {k:18s}: {v}")

    # ── Export Winning Configuration to YAML ─────────────────────────────────
    p = best_trial.params
    opt_type = p["optimizer"]
    if opt_type == "lion":
        best_lr = p["lr_lion"]
        best_wd = p["wd_lion"]
    elif opt_type == "adamw":
        best_lr = p["lr_adamw"]
        best_wd = p["wd_adamw"]
    elif opt_type == "adam":
        best_lr = p["lr_adam"]
        best_wd = p["wd_adam"]
    elif opt_type == "sgd":
        best_lr = p["lr_sgd"]
        best_wd = p["wd_sgd"]
    elif opt_type == "sam_adamw":
        best_lr = p["lr_sam"]
        best_wd = p["wd_sam"]
    else:
        best_lr = 1e-4
        best_wd = 1e-4

    # Pull both dataset path blocks from full.yaml so the exported config is
    # valid for either dataset (previously only `uwiqa` was written, even for saud).
    base_cfg_path = os.path.join(project_root, "configs", "full.yaml")
    with open(base_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    tuned_config = {
        "mode": "tuned",
        "dataset": dataset,
        "seed": args.seed,
        "image_size": 224,
        "patch_sizes": [32, 64, 128],
        "d_model": int(p["d_model"]),
        "nlayers": int(p["nlayers"]),
        "nhead": 8 if p["d_model"] == 512 else 4,
        "dropout": float(p["dropout"]),
        "epochs": 50,
        "min_epochs": 15,
        "batch_size": int(p["batch_size"]),
        "num_workers": args.workers,
        "early_stopping": {
            "monitor": "srocc",
            "mode": "max",
            "patience": 15,
            "min_delta": 0.001,
        },
        "optimizer": {
            "name": opt_type,
            "lr": float(best_lr),
            "weight_decay": float(best_wd),
        },
        "scheduler": p.get("scheduler", "none"),
        "loss": {
            "lambda_mos": 1.0,
            "lambda_rank": float(p["lambda_rank"]),
            "lambda_cons": float(p["lambda_cons"]),
            "margin": float(p["ranking_margin"]),
        },
        "augmentation": {
            "train": {
                "horizontal_flip": 0.5,
                "vertical_flip": float(p["vflip_prob"]),
                "rotation": int(p["rotation_deg"]),
            }
        },
        "uwiqa": base_cfg.get("uwiqa", {
            "image_dir": "../UWIQA dataset/UWIQA",
            "score_file": "../UWIQA dataset/IQA Value.xlsx",
        }),
        "saud": base_cfg.get("saud", {
            "benchmark_dir": "../SAUD/Benchmark",
            "bt_scores_mat": "../SAUD/sub_bt_scores.mat",
        }),
        "uid": base_cfg.get("uid", {
            "image_dir":  "../UID2021/UID2021",
            "score_file": "../mos_UID.xlsx",
        }),
        "checkpoint_dir": f"checkpoints/{dataset}/{args.study_name}",
        "checkpoint": {
            "save_best_on": "srocc",
            "mode": "max",
        },
        "device": args.device,
        "mixed_precision": True,
        "print_shapes": False,
    }

    out_cfg_path = os.path.join(project_root, args.output_config)
    os.makedirs(os.path.dirname(out_cfg_path), exist_ok=True)
    with open(out_cfg_path, "w") as f:
        yaml.dump(tuned_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n[Saved] Tuned configuration written to: {out_cfg_path}")

    # Save summary json
    summary_path = os.path.join(optuna_dir, f"{args.study_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "dataset": dataset,
                "best_trial_number": best_trial.number,
                "best_val_srocc": best_trial.value,
                "best_params": best_trial.params,
                "total_trials": len(study.trials),
                "duration_minutes": total_time / 60,
            },
            f,
            indent=2,
        )
    print(f"[Saved] Summary report written to: {summary_path}")

    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"1. To run final 10-iteration training with these tuned settings:")
    print(f"   python scripts/train.py --config {args.output_config} --iterations 10")
    print(f"2. To inspect the interactive Optuna web dashboard:")
    print(f"   optuna-dashboard {storage_url}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
