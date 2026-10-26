"""
scripts/train.py
----------------
Main entry point for HMST Architecture 1 training.

Usage:
    python scripts/train.py --mode test
    python scripts/train.py --mode full
    python scripts/train.py --mode full --resume checkpoints/full/latest.pt

The --mode flag selects configs/test.yaml or configs/full.yaml.
A --resume flag overrides config.resume and provides the checkpoint path.
"""

import sys
import os
import argparse
import random
import time
import json
import csv

# Make the project root importable from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

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
from training import HMSTTrainer, build_optimizer
from training.plots import (
    save_error_analysis_plots,
    save_model_interpretability_plots,
    save_quality_plots,
    save_split_distribution_plots,
    save_training_dynamics_plots,
)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """Auto-detect best available device or use config override."""
    dev_str = config.get("device", "auto")
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev_str)
    return device


def load_config(mode: str) -> dict:
    """Load YAML config for the given mode."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", f"{mode}.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_absolute(path: str, base: str) -> str:
    """Make a path absolute relative to base if it isn't already, with fallback resolution."""
    if os.path.isabs(path):
        resolved = path
    else:
        resolved = os.path.normpath(os.path.join(base, path))

    if os.path.exists(resolved):
        return resolved

    # Fallback 1: check upward parent directories (up to 4 levels)
    curr = base
    for _ in range(4):
        candidate = os.path.normpath(os.path.join(curr, path))
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    # Fallback 2: check without redundant parent directory (e.g. UWIQA/UWIQA dataset -> UWIQA dataset)
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[-2] == "UWIQA" and "UWIQA dataset" in parts[-1]:
        alt_path = path.replace("UWIQA/UWIQA dataset", "UWIQA dataset")
        alt_resolved = os.path.normpath(os.path.join(base, alt_path)) if not os.path.isabs(alt_path) else alt_path
        if os.path.exists(alt_resolved):
            return alt_resolved

    return resolved


def dataset_output_dir(dataset: str, mode: str, project_root: str, iter_suffix: str = "") -> str:
    """Dataset-scoped output directory: outputs/{dataset}/{mode}{iter_suffix}."""
    return make_absolute(f"outputs/{dataset}/{mode}{iter_suffix}", project_root)


def dataset_checkpoint_dir(dataset: str, mode: str, project_root: str, iter_suffix: str = "") -> str:
    """Dataset-scoped checkpoint directory: checkpoints/{dataset}/{mode}{iter_suffix}."""
    return make_absolute(f"checkpoints/{dataset}/{mode}{iter_suffix}", project_root)


def write_run_marker(directory: str, dataset: str, mode: str) -> None:
    """Write run.json identifying the dataset/mode that owns this directory."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "run.json"), "w") as f:
        json.dump({"dataset": dataset, "mode": mode}, f, indent=2)


def assert_dataset_matches(directory: str, dataset: str) -> None:
    """Hard-fail if an existing run.json in `directory` belongs to a different dataset."""
    marker = os.path.join(directory, "run.json")
    if not os.path.exists(marker):
        return
    with open(marker, "r") as f:
        recorded = json.load(f)
    if recorded.get("dataset") != dataset:
        raise RuntimeError(
            f"Dataset mismatch in '{directory}': run.json records dataset "
            f"'{recorded.get('dataset')}' but current dataset is '{dataset}'. "
            f"Refusing to mix artifacts from different datasets."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(config: dict, project_root: str):
    """Build train/val/test datasets from config."""
    dataset_name = config.get("dataset", "uwiqa")
    image_size   = config.get("image_size", 224)

    aug_cfg  = config.get("augmentation", {}).get("train", {})
    train_tf = get_train_transforms(
        image_size = image_size,
        hflip_p    = aug_cfg.get("horizontal_flip", 0.5),
        vflip_p    = aug_cfg.get("vertical_flip", 0.5),
        degrees    = aug_cfg.get("rotation", 10.0),
    )
    val_tf   = get_val_transforms(image_size)

    max_train = config.get("max_train_samples")
    max_val   = config.get("max_val_samples")
    max_test  = config.get("max_test_samples")
    seed      = config.get("seed", 42)

    if dataset_name == "uwiqa":
        cfg = config["uwiqa"]
        image_dir  = make_absolute(cfg["image_dir"],  project_root)
        score_file = make_absolute(cfg["score_file"], project_root)

        train_idx, test_idx = make_uwiqa_splits(image_dir, score_file, seed=seed)

        # Val = last 20% of train indices
        val_size = max(1, len(train_idx) // 5)
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = UWIQADataset(image_dir, score_file, train_idx, train_tf, max_train)
        val_ds   = UWIQADataset(image_dir, score_file, val_idx,   val_tf,   max_val)
        test_ds  = UWIQADataset(image_dir, score_file, test_idx,  val_tf,   max_test)

    elif dataset_name == "saud":
        cfg = config["saud"]
        benchmark_dir = make_absolute(cfg["benchmark_dir"], project_root)
        bt_mat        = make_absolute(cfg["bt_scores_mat"], project_root)

        train_idx, test_idx = make_saud_splits(bt_mat, seed=seed)

        val_size  = max(1, len(train_idx) // 5)
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = SAUDDataset(benchmark_dir, bt_mat, train_idx, train_tf, max_train)
        val_ds   = SAUDDataset(benchmark_dir, bt_mat, val_idx,   val_tf,   max_val)
        test_ds  = SAUDDataset(benchmark_dir, bt_mat, test_idx,  val_tf,   max_test)

    elif dataset_name == "uied":
        cfg = config["uied"]
        image_dir  = make_absolute(cfg["image_dir"], project_root)
        score_file = make_absolute(cfg["score_file"], project_root)

        # Import locally to avoid top-level issues if not present
        from data import UIEDDataset, make_uied_splits
        
        train_idx, test_idx = make_uied_splits(score_file, seed=seed)

        val_size  = max(1, len(train_idx) // 5)
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = UIEDDataset(image_dir, score_file, train_idx, train_tf, max_train)
        val_ds   = UIEDDataset(image_dir, score_file, val_idx,   val_tf,   max_val)
        test_ds  = UIEDDataset(image_dir, score_file, test_idx,  val_tf,   max_test)

    elif dataset_name == "uid":
        cfg = config["uid"]
        image_dir  = make_absolute(cfg["image_dir"],  project_root)
        score_file = make_absolute(cfg["score_file"], project_root)

        train_idx, test_idx = make_uid_splits(image_dir, score_file, seed=seed)

        val_size  = max(16, (len(train_idx) // 5) // 16 * 16)   # whole scenes of 16 images
        val_idx   = train_idx[-val_size:]
        train_idx = train_idx[:-val_size]

        train_ds = UIDDataset(image_dir, score_file, train_idx, train_tf, max_train)
        val_ds   = UIDDataset(image_dir, score_file, val_idx,   val_tf,   max_val)
        test_ds  = UIDDataset(image_dir, score_file, test_idx,  val_tf,   max_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, val_ds, test_ds


def build_full_evaluation_dataset(config: dict, project_root: str):
    """Build the complete, non-augmented UWIQA dataset for per-image plots."""
    if config.get("dataset", "uwiqa") != "uwiqa":
        return None

    dataset_config = config["uwiqa"]
    image_dir = make_absolute(dataset_config["image_dir"], project_root)
    score_file = make_absolute(dataset_config["score_file"], project_root)
    train_indices, test_indices = make_uwiqa_splits(
        image_dir, score_file, seed=config.get("seed", 42)
    )
    all_indices = sorted(train_indices + test_indices)
    return UWIQADataset(
        image_dir,
        score_file,
        all_indices,
        get_val_transforms(config.get("image_size", 224)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HMST — Architecture 1")
    parser.add_argument(
        "--mode", type=str, choices=["test", "full"], default="test",
        help="Select configs/test.yaml or configs/full.yaml"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom config YAML file (e.g. configs/uwiqa_tuned.yaml). Overrides --mode"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to resume from (overrides config.resume)"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, choices=["uwiqa", "saud", "uied", "uid"],
        help="Override dataset from config"
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of iterations to run with random splits"
    )
    parser.add_argument(
        "--rolling-seeds", action="store_true", default=True,
        help="Automatically roll seeds across iterations (enabled by default)"
    )
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

    # ── Config ────────────────────────────────────────────────────────────
    if args.config:
        if os.path.isabs(args.config):
            cfg_path = args.config
        elif os.path.exists(args.config):
            cfg_path = os.path.abspath(args.config)
        elif os.path.exists(os.path.join(project_root, args.config)):
            cfg_path = os.path.join(project_root, args.config)
        elif os.path.exists(os.path.join(project_root, "configs", args.config)):
            cfg_path = os.path.join(project_root, "configs", args.config)
        else:
            cfg_path = os.path.join(project_root, args.config)

        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.mode)
    if args.dataset:
        config["dataset"] = args.dataset

    mode = config.get("mode", args.mode or "full")
    dataset = config.get("dataset", "uwiqa")

    # Override resume from CLI
    resume_path = args.resume
    if resume_path is None and config.get("resume", False):
        default_resume_dir = dataset_checkpoint_dir(dataset, mode, project_root)
        assert_dataset_matches(default_resume_dir, dataset)
        default_resume = os.path.join(default_resume_dir, "latest.pt")
        if os.path.exists(default_resume):
            resume_path = default_resume

    # ── Iterations Loop ───────────────────────────────────────────────────
    num_iterations = getattr(args, "iterations", 1)
    base_seed = config.get("seed", 42)
    
    if num_iterations > 1:
        seed_print = f"Rolling (Base: {base_seed})"
    else:
        seed_print = str(base_seed) if base_seed is not None else "Random (None)"

    # ── Device ────────────────────────────────────────────────────────────
    device = get_device(config)
    print(f"\n{'='*60}")
    print(f"HMST Architecture 1 — Training")
    print(f"{'='*60}")
    print(f"Mode:     {mode}")
    print(f"Dataset:  {config.get('dataset', 'uwiqa')}")
    print(f"Device:   {device}")
    if device.type == "cuda":
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Seed:     {seed_print}")

    all_metrics = []
    all_metrics_no_tta = []

    for it in range(num_iterations):
        iter_suffix = f"_iter{it+1}" if num_iterations > 1 else ""
        checkpoint_dir = dataset_checkpoint_dir(dataset, mode, project_root, iter_suffix)
        output_dir = dataset_output_dir(dataset, mode, project_root, iter_suffix)
        # Hard-fail if either dir was previously owned by a different dataset.
        assert_dataset_matches(checkpoint_dir, dataset)
        assert_dataset_matches(output_dir, dataset)
        write_run_marker(checkpoint_dir, dataset, mode)
        write_run_marker(output_dir, dataset, mode)
        final_path = os.path.join(output_dir, "test_metrics.json")
        std_path = os.path.join(output_dir, "test_metrics_no_tta.json")

        required_plot_names = (
            "A1_test_predicted_vs_actual_tta.png",
            "A2_test_raw_vs_fitted.png",
            "A3_validation_vs_test_tta.png",
            "A4_test_error_magnitude.png",
            "B5_residual_vs_actual_mos.png",
            "B6_error_histogram_kde_tta_comparison.png",
            "B7_absolute_error_violin_box.png",
            "B8_best_worst_prediction_images.png",
            "B9_per_image_error_pareto.png",
            "C10_loss_curves.png",
            "C11_metric_curves.png",
            "C12_learning_rate_schedule.png",
            "C13_early_stopping_convergence.png",
            "F22_mos_distribution.png",
            "F23_split_sample_counts.png",
            "F24_split_distribution_ks.png",
        )
        history_path = os.path.join(output_dir, "history.csv")
        training_plot_paths = [
            os.path.join(output_dir, "plots", name)
            for name in required_plot_names
            if name.startswith("C")
        ]
        if os.path.isfile(history_path) and not all(
            os.path.isfile(path) for path in training_plot_paths
        ):
            with open(history_path, "r", newline="", encoding="utf-8") as history_file:
                history = [
                    {key: float(value) for key, value in row.items() if value not in (None, "")}
                    for row in csv.DictReader(history_file)
                ]
            save_training_dynamics_plots(
                history,
                output_dir=os.path.join(output_dir, "plots"),
                monitor=config.get("early_stopping", {}).get("monitor", "srocc"),
                mode=config.get("early_stopping", {}).get("mode", "max"),
            )
        split_plot_paths = [
            os.path.join(output_dir, "plots", name)
            for name in required_plot_names
            if name.startswith("F")
        ]
        if os.path.exists(final_path) and not all(
            os.path.isfile(path) for path in split_plot_paths
        ):
            split_config = dict(config)
            split_config["seed"] = base_seed + it if base_seed is not None else 42000 + it
            split_train_ds, split_val_ds, split_test_ds = build_datasets(
                split_config, project_root
            )
            save_split_distribution_plots(
                split_train_ds,
                split_val_ds,
                split_test_ds,
                output_dir=os.path.join(output_dir, "plots"),
                mos_range=(0.2, 1.0) if dataset == "uwiqa" else (0.0, 1.0),
            )
        plots_complete = all(
            os.path.isfile(os.path.join(output_dir, "plots", name))
            for name in required_plot_names
            )
        if num_iterations > 1 and os.path.exists(final_path) and plots_complete:
            print(f"\n[Trainer] Iteration {it+1} already completed. Loading metrics and skipping...")
            import json
            with open(final_path, "r") as f:
                loaded_tta = json.load(f)
            loaded_tta.setdefault("iteration", it + 1)
            if base_seed is not None:
                loaded_tta.setdefault("seed", base_seed + it)
            all_metrics.append(loaded_tta)
            if os.path.exists(std_path):
                with open(std_path, "r") as f:
                    loaded_no_tta = json.load(f)
                loaded_no_tta.setdefault("iteration", it + 1)
                if base_seed is not None:
                    loaded_no_tta.setdefault("seed", base_seed + it)
                all_metrics_no_tta.append(loaded_no_tta)
            continue
        if num_iterations > 1 and os.path.exists(final_path) and not plots_complete:
            print(
                f"\n[Trainer] Iteration {it+1} has metrics but is missing correlation plots; "
                "running its evaluation path to create them."
            )

        if num_iterations > 1:
            print(f"\n{'#'*60}")
            print(f" ITERATION {it+1}/{num_iterations}")
            print(f"{'#'*60}")
            
        # Predictable seed for resuming dataset splits exactly
        iter_seed = base_seed + it if base_seed is not None else 42000 + it
        config["seed"] = iter_seed
        set_seed(iter_seed)
        
        # ── Datasets ──────────────────────────────────────────────────────────
        print(f"\nBuilding datasets...")
        train_ds, val_ds, test_ds = build_datasets(config, project_root)
        save_split_distribution_plots(
            train_ds,
            val_ds,
            test_ds,
            output_dir=os.path.join(output_dir, "plots"),
            mos_range=(0.2, 1.0) if dataset == "uwiqa" else (0.0, 1.0),
        )

        print(f"  Train:  {len(train_ds)} samples")
        print(f"  Val:    {len(val_ds)} samples")
        print(f"  Test:   {len(test_ds)} samples")

        num_workers = config.get("num_workers", 0)
        batch_size  = config.get("batch_size", 4)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            drop_last=(len(train_ds) > batch_size),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

        # ── Model ─────────────────────────────────────────────────────────────
        print(f"\nBuilding HMST model...")
        patch_sizes  = config.get("patch_sizes", [32, 64, 128])
        image_size   = config.get("image_size", 224)
        print_shapes = config.get("print_shapes", False)

        d_model   = int(config.get("d_model", 512))
        nhead     = int(config.get("nhead", 8))
        nlayers   = int(config.get("nlayers", 2))
        dropout   = float(config.get("dropout", 0.3))
        n_regions = int(config.get("n_regions", 4))

        model = HMST(
            image_size   = image_size,
            patch_sizes  = patch_sizes,
            d_model      = d_model,
            nhead        = nhead,
            nlayers      = nlayers,
            n_regions    = n_regions,
            dropout      = dropout,
            print_shapes = print_shapes if it == 0 else False,
        ).to(device)

        if it == 0:
            total_params     = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Total parameters:     {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")

        # ── Loss (§4.9) ──────────────────────────────────────────────────────
        print(f"\nBuilding loss function...")
        M = build_correspondence_matrix(
            num_patches_fine   = model.num_patches[0],   # 49 (p=32)
            num_patches_coarse = model.num_patches[2],   # 1  (p=128)
            grid_fine          = model.grid_sizes[0],    # 7
            grid_coarse        = model.grid_sizes[2],    # 1
        ).to(device)

        loss_cfg = config.get("loss", {})
        criterion = HMSTLoss(
            M            = M,
            lambda_mos   = loss_cfg.get("lambda_mos",  1.0),
            lambda_rank  = loss_cfg.get("lambda_rank", 1.0),
            lambda_cons  = loss_cfg.get("lambda_cons", 0.1),
            margin       = loss_cfg.get("margin", 0.1),
        )

        # ── Optimizer ───────────────────────────────────────────────────────────
        opt_cfg = config.get("optimizer", {})
        opt_name = opt_cfg.get("name", "adam").lower()
        lr        = float(opt_cfg.get("lr", 1e-4))
        wd        = float(opt_cfg.get("weight_decay", 1e-4))
        opt_extra = {k: v for k, v in opt_cfg.items() if k not in ("name", "lr", "weight_decay")}
        optimizer = build_optimizer(model, opt_name, lr=lr, weight_decay=wd, **opt_extra)

        # ── Scheduler ─────────────────────────────────────────────────────────
        sched_name = config.get("scheduler", "none")
        scheduler = None
        if sched_name == "cosine":
            total_epochs = config.get("epochs", 50)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=lr * 0.01
            )
        elif sched_name == "step":
            total_epochs = config.get("epochs", 50)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(5, total_epochs // 3), gamma=0.5
            )

        # ── Trainer ───────────────────────────────────────────────────────────
        trainer = HMSTTrainer(
            model          = model,
            criterion      = criterion,
            optimizer      = optimizer,
            config         = config,
            device         = device,
            output_dir     = output_dir,
            checkpoint_dir = checkpoint_dir,
            scheduler      = scheduler,
        )

        # ── Shape validation (test mode) ──────────────────────────────────────
        if mode == "test" and it == 0:
            print(f"\n{'─'*60}")
            print("Running architecture shape validation (test mode)...")
            model.print_shapes = False
            dummy = torch.zeros(2, 3, image_size, image_size).to(device)
            with torch.no_grad():
                out = model(dummy)
            print(f"Output Q_hat shape: {list(out['Q_hat'].shape)}")
            assert out["Q_hat"].shape == (2,), f"Expected [2], got {out['Q_hat'].shape}"
            print("Shape validation: PASS")
            print(f"{'─'*60}\n")

        model.print_shapes = False

        # ── Train ─────────────────────────────────────────────────────────────
        iter_resume_path = None
        if config.get("resume", False):
            default_iter_resume = os.path.join(checkpoint_dir, "latest.pt")
            if os.path.exists(default_iter_resume):
                iter_resume_path = default_iter_resume
                
        if args.resume and it == 0:
            iter_resume_path = args.resume

        trainer.fit(train_loader, val_loader, resume_path=iter_resume_path)

        # ── Final test evaluation ─────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Final evaluation on test set for iteration {it+1}...")
        from training.metrics import (
            compute_metrics,
            collect_interpretability_data,
            collect_predictions,
            collect_predictions_tta,
        )

        best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(best_ckpt_path):
            from training.checkpoint import load_checkpoint
            load_checkpoint(best_ckpt_path, model, device=device)
            print(f"Loaded best.pt for final evaluation")

        # ── Standard single-pass evaluation ───────────────────────────────────
        result = collect_predictions(model, test_loader, device)
        final_metrics = compute_metrics(result["predictions"], result["targets"])

        # ── TTA evaluation (4-view flip average) ──────────────────────────────
        print(f"Running TTA evaluation (4 flip views)...")
        result_tta = collect_predictions_tta(model, test_loader, device)
        final_metrics_tta = compute_metrics(result_tta["predictions"], result_tta["targets"])

        if dataset in {"uwiqa", "saud", "uid"}:
            validation_result_tta = collect_predictions_tta(model, val_loader, device)
            validation_metrics_tta = compute_metrics(
                validation_result_tta["predictions"],
                validation_result_tta["targets"],
            )
            plot_paths = save_quality_plots(
                actual_test=result_tta["targets"],
                predicted_test_tta=result_tta["predictions"],
                test_metrics=final_metrics_tta,
                actual_validation=validation_result_tta["targets"],
                predicted_validation_tta=validation_result_tta["predictions"],
                validation_metrics=validation_metrics_tta,
                output_dir=os.path.join(output_dir, "plots"),
            )
            pareto_dataset = build_full_evaluation_dataset(config, project_root)
            pareto_result = result_tta
            if pareto_dataset is not None:
                pareto_loader = DataLoader(
                    pareto_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=(device.type == "cuda"),
                )
                pareto_result = collect_predictions_tta(model, pareto_loader, device)
            error_plot_paths = save_error_analysis_plots(
                actual=result_tta["targets"],
                predicted_tta=result_tta["predictions"],
                predicted_no_tta=result["predictions"],
                image_paths=result_tta.get("paths", []),
                output_dir=os.path.join(output_dir, "plots"),
                pareto_actual=pareto_result["targets"],
                pareto_predicted=pareto_result["predictions"],
            )
            interpretability_data = collect_interpretability_data(model, test_loader, device)
            interpretability_plot_paths = save_model_interpretability_plots(
                interpretability_data,
                model,
                M,
                output_dir=os.path.join(output_dir, "plots"),
            )
            print("Correlation plots saved:")
            for plot_path in (*plot_paths, *error_plot_paths, *interpretability_plot_paths):
                print(f"  {plot_path}")

        # Tag metrics with iteration index / seed (used for best-iteration tracking)
        final_metrics_tta["iteration"] = it + 1
        final_metrics_tta["seed"] = iter_seed
        final_metrics["iteration"] = it + 1
        final_metrics["seed"] = iter_seed

        # Use TTA metrics as the reported result (better stability)
        all_metrics.append(final_metrics_tta)
        all_metrics_no_tta.append(final_metrics)

        # Persist MAE / MedAE into the metric dicts (previously console-only)
        abs_err_tta = np.abs(result_tta["predictions"] - result_tta["targets"])
        abs_err_no_tta = np.abs(result["predictions"] - result["targets"])
        final_metrics_tta.update({
            "mae": float(np.mean(abs_err_tta)),
            "medae": float(np.median(abs_err_tta)),
        })
        final_metrics.update({
            "mae": float(np.mean(abs_err_no_tta)),
            "medae": float(np.median(abs_err_no_tta)),
        })

        import json
        final_path = os.path.join(output_dir, "test_metrics.json")
        with open(final_path, "w") as f:
            json.dump(final_metrics_tta, f, indent=2)

        # Also save standard metrics for reference
        std_path = os.path.join(output_dir, "test_metrics_no_tta.json")
        with open(std_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        # ── Per-image predictions CSV (Actual vs Predicted analysis) ──────────
        import csv
        pred_csv_path = os.path.join(output_dir, "test_predictions.csv")
        paths = result_tta.get("paths", result.get("paths", []))
        preds_no_tta = result["predictions"]
        preds_tta = result_tta["predictions"]
        actuals = result_tta["targets"]

        rows = []
        for idx in range(len(actuals)):
            full_path = paths[idx] if idx < len(paths) else f"test_{idx:04d}"
            img_name = os.path.basename(full_path)
            act = float(actuals[idx])
            p_tta = float(preds_tta[idx])
            p_raw = float(preds_no_tta[idx])
            err_tta = p_tta - act
            abs_err = abs(err_tta)
            rows.append({
                "image": img_name,
                "actual_mos": act,
                "pred_mos_tta": p_tta,
                "pred_mos_no_tta": p_raw,
                "abs_error_tta": abs_err,
                "error_tta": err_tta,
                "path": full_path,
            })

        with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "actual_mos", "pred_mos_tta", "pred_mos_no_tta", "abs_error_tta", "error_tta", "path"])
            for r in rows:
                writer.writerow([
                    r["image"],
                    f"{r['actual_mos']:.4f}",
                    f"{r['pred_mos_tta']:.4f}",
                    f"{r['pred_mos_no_tta']:.4f}",
                    f"{r['abs_error_tta']:.4f}",
                    f"{r['error_tta']:.4f}",
                    r["path"],
                ])

        print(f"\nTest Set Results (Iter {it+1}):")
        print(f"  {'Metric':<8}  {'No TTA':>8}  {'TTA (4-view)':>12}")
        print(f"  {'-'*34}")
        print(f"  {'PLCC':<8}  {final_metrics['plcc']:>8.4f}  {final_metrics_tta['plcc']:>12.4f}")
        print(f"  {'SROCC':<8}  {final_metrics['srocc']:>8.4f}  {final_metrics_tta['srocc']:>12.4f}")
        print(f"  {'KRCC':<8}  {final_metrics['krcc']:>8.4f}  {final_metrics_tta['krcc']:>12.4f}")
        print(f"  {'RMSE':<8}  {final_metrics['rmse']:>8.4f}  {final_metrics_tta['rmse']:>12.4f}")
        print(f"TTA results saved to: {final_path}")
        print(f"Per-image predictions saved to: {pred_csv_path}")

        # Quick console highlight: top 3 best and top 3 worst, plus MAE & MedAE
        sorted_by_err = sorted(rows, key=lambda x: x["abs_error_tta"])
        best_3 = sorted_by_err[:min(3, len(sorted_by_err))]
        worst_3 = sorted_by_err[-min(3, len(sorted_by_err)):][::-1]
        all_abs_errors = [r["abs_error_tta"] for r in rows]
        mean_ae = float(np.mean(all_abs_errors))
        med_ae = float(np.median(all_abs_errors))

        print(f"\n  Per-Image Prediction Highlights (Iter {it+1}):")
        print(f"    Mean Absolute Error (MAE):     {mean_ae:.4f}")
        print(f"    Median Absolute Error (MedAE):  {med_ae:.4f}")
        print(f"    Closest Predictions (Best):")
        for b in best_3:
            print(f"      • {b['image']:<12}: Actual={b['actual_mos']:.4f}, Pred={b['pred_mos_tta']:.4f} (Diff: {b['abs_error_tta']:.4f})")
        print(f"    Largest Prediction Errors (Failure Cases):")
        for w in worst_3:
            print(f"      • {w['image']:<12}: Actual={w['actual_mos']:.4f}, Pred={w['pred_mos_tta']:.4f} (Diff: {w['abs_error_tta']:.4f})")
        print(f"{'='*60}\n")

    if len(all_metrics) > 0:
        summary_dir = dataset_output_dir(dataset, mode, project_root)
        os.makedirs(summary_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"BENCHMARK SUMMARY ACROSS {len(all_metrics)} ITERATIONS (Single View vs TTA)")
        print(f"{'='*65}")

        plccs  = [m["plcc"] for m in all_metrics]
        sroccs = [m["srocc"] for m in all_metrics]
        krccs  = [m["krcc"] for m in all_metrics]
        rmses  = [m["rmse"] for m in all_metrics]

        med_plcc,  mean_plcc,  std_plcc  = float(np.median(plccs)),  float(np.mean(plccs)),  float(np.std(plccs))
        med_srocc, mean_srocc, std_srocc = float(np.median(sroccs)), float(np.mean(sroccs)), float(np.std(sroccs))
        med_krcc,  mean_krcc,  std_krcc  = float(np.median(krccs)),  float(np.mean(krccs)),  float(np.std(krccs))
        med_rmse,  mean_rmse,  std_rmse  = float(np.median(rmses)),  float(np.mean(rmses)),  float(np.std(rmses))

        has_no_tta = (len(all_metrics_no_tta) == len(all_metrics))
        if has_no_tta:
            plccs_raw  = [m["plcc"] for m in all_metrics_no_tta]
            sroccs_raw = [m["srocc"] for m in all_metrics_no_tta]
            krccs_raw  = [m["krcc"] for m in all_metrics_no_tta]
            rmses_raw  = [m["rmse"] for m in all_metrics_no_tta]

            med_plcc_raw,  mean_plcc_raw,  std_plcc_raw  = float(np.median(plccs_raw)),  float(np.mean(plccs_raw)),  float(np.std(plccs_raw))
            med_srocc_raw, mean_srocc_raw, std_srocc_raw = float(np.median(sroccs_raw)), float(np.mean(sroccs_raw)), float(np.std(sroccs_raw))
            med_krcc_raw,  mean_krcc_raw,  std_krcc_raw  = float(np.median(krccs_raw)),  float(np.mean(krccs_raw)),  float(np.std(krccs_raw))
            med_rmse_raw,  mean_rmse_raw,  std_rmse_raw  = float(np.median(rmses_raw)),  float(np.mean(rmses_raw)),  float(np.std(rmses_raw))

            print(f"  {'Metric':<8}  {'Single View (No TTA)':<25}  {'TTA (4-view)':<25}")
            print(f"  {'-'*62}")
            print(f"  {'SROCC:':<8}  {med_srocc_raw:.4f} (Mean: {mean_srocc_raw:.4f}±{std_srocc_raw:.4f})   {med_srocc:.4f} (Mean: {mean_srocc:.4f}±{std_srocc:.4f})")
            print(f"  {'PLCC:':<8}  {med_plcc_raw:.4f} (Mean: {mean_plcc_raw:.4f}±{std_plcc_raw:.4f})   {med_plcc:.4f} (Mean: {mean_plcc:.4f}±{std_plcc:.4f})")
            print(f"  {'KRCC:':<8}  {med_krcc_raw:.4f} (Mean: {mean_krcc_raw:.4f}±{std_krcc_raw:.4f})   {med_krcc:.4f} (Mean: {mean_krcc:.4f}±{std_krcc:.4f})")
            print(f"  {'RMSE:':<8}  {med_rmse_raw:.4f} (Mean: {mean_rmse_raw:.4f}±{std_rmse_raw:.4f})   {med_rmse:.4f} (Mean: {mean_rmse:.4f}±{std_rmse:.4f})")
            print(f"{'='*65}\n")
        else:
            print(f"  Metric       Median       Mean        Std")
            print(f"  ---------------------------------------------")
            print(f"  SROCC:       {med_srocc:.4f}       {mean_srocc:.4f}      (±{std_srocc:.4f})")
            print(f"  PLCC:        {med_plcc:.4f}       {mean_plcc:.4f}      (±{std_plcc:.4f})  [4-param logistic]")
            print(f"  KRCC:        {med_krcc:.4f}       {mean_krcc:.4f}      (±{std_krcc:.4f})")
            print(f"  RMSE:        {med_rmse:.4f}       {mean_rmse:.4f}      (±{std_rmse:.4f})  [4-param logistic]")
            print(f"{'='*60}\n")

        summary_data = {
            "num_iterations": len(all_metrics),
            "tta_4view": {
                "median": {"srocc": med_srocc, "plcc": med_plcc, "krcc": med_krcc, "rmse": med_rmse},
                "mean":   {"srocc": mean_srocc, "plcc": mean_plcc, "krcc": mean_krcc, "rmse": mean_rmse},
                "std":    {"srocc": std_srocc, "plcc": std_plcc, "krcc": std_krcc, "rmse": std_rmse},
                "all_iterations": all_metrics,
            },
        }
        if has_no_tta:
            summary_data["single_view_no_tta"] = {
                "median": {"srocc": med_srocc_raw, "plcc": med_plcc_raw, "krcc": med_krcc_raw, "rmse": med_rmse_raw},
                "mean":   {"srocc": mean_srocc_raw, "plcc": mean_plcc_raw, "krcc": mean_krcc_raw, "rmse": mean_rmse_raw},
                "std":    {"srocc": std_srocc_raw, "plcc": std_plcc_raw, "krcc": std_krcc_raw, "rmse": std_rmse_raw},
                "all_iterations": all_metrics_no_tta,
            }

        # ── Best iteration (selected by TTA SROCC) ────────────────────────────
        best_idx = int(np.argmax([m["srocc"] for m in all_metrics]))
        best_metric = all_metrics[best_idx]
        best_iter = int(best_metric.get("iteration", best_idx + 1))
        summary_data["best_iteration"] = {
            "criterion": "srocc (TTA 4-view)",
            "iteration": best_iter,
            "seed": best_metric.get("seed"),
            "metrics": best_metric,
        }
        print(f"{'='*65}")
        print(f"BEST ITERATION (by TTA SROCC): Iteration {best_iter}  (seed={best_metric.get('seed')})")
        print(f"{'='*65}")
        print(f"  SROCC: {best_metric['srocc']:.4f}  PLCC: {best_metric['plcc']:.4f}  "
              f"KRCC: {best_metric['krcc']:.4f}  RMSE: {best_metric['rmse']:.4f}")
        if "mae" in best_metric:
            print(f"  MAE:   {best_metric['mae']:.4f}  MedAE: {best_metric.get('medae', float('nan')):.4f}")

        # Persist consolidated copies of the best iteration's artifacts
        best_iter_suffix = f"_iter{best_iter}" if num_iterations > 1 else ""
        best_src_dir = dataset_output_dir(dataset, mode, project_root, best_iter_suffix)
        import shutil
        best_pred_src = os.path.join(best_src_dir, "test_predictions.csv")
        best_pred_dst = os.path.join(summary_dir, "best_iteration_predictions.csv")
        if os.path.exists(best_pred_src) and os.path.abspath(best_pred_src) != os.path.abspath(best_pred_dst):
            shutil.copyfile(best_pred_src, best_pred_dst)
            print(f"  Best per-image predictions -> {best_pred_dst}")
        best_metric_dst = os.path.join(summary_dir, "best_iteration_metrics.json")
        with open(best_metric_dst, "w") as f:
            json.dump(best_metric, f, indent=2)
        print(f"  Best metrics JSON -> {best_metric_dst}")
        print(f"{'='*65}\n")

        # ── Top-10 Summary (when running > 10 iterations, e.g. 15) ──────────
        if len(all_metrics) > 10:
            top_k = 10

            # 1. Top-k for TTA
            sorted_by_srocc = sorted(all_metrics, key=lambda m: m["srocc"], reverse=True)
            top_metrics = sorted_by_srocc[:top_k]

            t_plccs  = [m["plcc"] for m in top_metrics]
            t_sroccs = [m["srocc"] for m in top_metrics]
            t_krccs  = [m["krcc"] for m in top_metrics]
            t_rmses  = [m["rmse"] for m in top_metrics]

            t_med_plcc,  t_mean_plcc,  t_std_plcc  = float(np.median(t_plccs)),  float(np.mean(t_plccs)),  float(np.std(t_plccs))
            t_med_srocc, t_mean_srocc, t_std_srocc = float(np.median(t_sroccs)), float(np.mean(t_sroccs)), float(np.std(t_sroccs))
            t_med_krcc,  t_mean_krcc,  t_std_krcc  = float(np.median(t_krccs)),  float(np.mean(t_krccs)),  float(np.std(t_krccs))
            t_med_rmse,  t_mean_rmse,  t_std_rmse  = float(np.median(t_rmses)),  float(np.mean(t_rmses)),  float(np.std(t_rmses))

            summary_data["top_10"] = {
                "tta_4view": {
                    "median": {"srocc": t_med_srocc, "plcc": t_med_plcc, "krcc": t_med_krcc, "rmse": t_med_rmse},
                    "mean":   {"srocc": t_mean_srocc, "plcc": t_mean_plcc, "krcc": t_mean_krcc, "rmse": t_mean_rmse},
                    "std":    {"srocc": t_std_srocc, "plcc": t_std_plcc, "krcc": t_std_krcc, "rmse": t_std_rmse},
                    "top_iterations": top_metrics,
                }
            }

            if has_no_tta and len(all_metrics_no_tta) > 10:
                # 2. Top-k for Single View (No TTA)
                sorted_by_srocc_raw = sorted(all_metrics_no_tta, key=lambda m: m["srocc"], reverse=True)
                top_metrics_raw = sorted_by_srocc_raw[:top_k]

                t_plccs_raw  = [m["plcc"] for m in top_metrics_raw]
                t_sroccs_raw = [m["srocc"] for m in top_metrics_raw]
                t_krccs_raw  = [m["krcc"] for m in top_metrics_raw]
                t_rmses_raw  = [m["rmse"] for m in top_metrics_raw]

                t_med_plcc_raw,  t_mean_plcc_raw,  t_std_plcc_raw  = float(np.median(t_plccs_raw)),  float(np.mean(t_plccs_raw)),  float(np.std(t_plccs_raw))
                t_med_srocc_raw, t_mean_srocc_raw, t_std_srocc_raw = float(np.median(t_sroccs_raw)), float(np.mean(t_sroccs_raw)), float(np.std(t_sroccs_raw))
                t_med_krcc_raw,  t_mean_krcc_raw,  t_std_krcc_raw  = float(np.median(t_krccs_raw)),  float(np.mean(t_krccs_raw)),  float(np.std(t_krccs_raw))
                t_med_rmse_raw,  t_mean_rmse_raw,  t_std_rmse_raw  = float(np.median(rmses_raw)),  float(np.mean(rmses_raw)),  float(np.std(rmses_raw))

                summary_data["top_10"]["single_view_no_tta"] = {
                    "median": {"srocc": t_med_srocc_raw, "plcc": t_med_plcc_raw, "krcc": t_med_krcc_raw, "rmse": t_med_rmse_raw},
                    "mean":   {"srocc": t_mean_srocc_raw, "plcc": t_mean_plcc_raw, "krcc": t_mean_krcc_raw, "rmse": t_mean_rmse_raw},
                    "std":    {"srocc": t_std_srocc_raw, "plcc": t_std_plcc_raw, "krcc": t_std_krcc_raw, "rmse": t_std_rmse_raw},
                    "top_iterations": top_metrics_raw,
                }

                print(f"{'='*65}")
                print(f"TOP {top_k} BEST ITERATIONS SUMMARY (Ranked by SROCC out of {len(all_metrics)})")
                print(f"{'='*65}")
                print(f"  {'Metric':<8}  {'Single View (No TTA)':<25}  {'TTA (4-view)':<25}")
                print(f"  {'-'*62}")
                print(f"  {'SROCC:':<8}  {t_med_srocc_raw:.4f} (Mean: {t_mean_srocc_raw:.4f}±{t_std_srocc_raw:.4f})   {t_med_srocc:.4f} (Mean: {t_mean_srocc:.4f}±{t_std_srocc:.4f})")
                print(f"  {'PLCC:':<8}  {t_med_plcc_raw:.4f} (Mean: {t_mean_plcc_raw:.4f}±{t_std_plcc_raw:.4f})   {t_med_plcc:.4f} (Mean: {t_mean_plcc:.4f}±{t_std_plcc:.4f})")
                print(f"  {'KRCC:':<8}  {t_med_krcc_raw:.4f} (Mean: {t_mean_krcc_raw:.4f}±{t_std_krcc_raw:.4f})   {t_med_krcc:.4f} (Mean: {t_mean_krcc:.4f}±{t_std_krcc:.4f})")
                print(f"  {'RMSE:':<8}  {t_med_rmse_raw:.4f} (Mean: {t_mean_rmse_raw:.4f}±{t_std_rmse_raw:.4f})   {t_med_rmse:.4f} (Mean: {t_mean_rmse:.4f}±{t_std_rmse:.4f})")
                print(f"{'='*65}\n")
            else:
                print(f"{'='*60}")
                print(f"TOP {top_k} BEST ITERATIONS SUMMARY (Ranked by SROCC out of {len(all_metrics)})")
                print(f"{'='*60}")
                print(f"  Metric       Median       Mean        Std")
                print(f"  ---------------------------------------------")
                print(f"  SROCC:       {t_med_srocc:.4f}       {t_mean_srocc:.4f}      (±{t_std_srocc:.4f})")
                print(f"  PLCC:        {t_med_plcc:.4f}       {t_mean_plcc:.4f}      (±{t_std_plcc:.4f})  [4-param logistic]")
                print(f"  KRCC:        {t_med_krcc:.4f}       {t_mean_krcc:.4f}      (±{t_std_krcc:.4f})")
                print(f"  RMSE:        {t_med_rmse:.4f}       {t_mean_rmse:.4f}      (±{t_std_rmse:.4f})  [4-param logistic]")
                print(f"{'='*60}\n")

        summary_file = os.path.join(summary_dir, f"{num_iterations}_iterations_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Full benchmark summary saved to: {summary_file}\n")


if __name__ == "__main__":
    main()
