"""
scripts/evaluate.py
--------------------
Evaluate a trained HMST model on the test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/full/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/test/best.pt --mode test
"""

import sys
import os
import argparse
import json
import csv

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
import yaml

from scripts.train import (
    load_config,
    build_datasets,
    get_device,
    make_absolute,
    set_seed,
    dataset_output_dir,
    build_full_evaluation_dataset,
)
from models import HMST, build_correspondence_matrix
from models.losses import HMSTLoss
from training.checkpoint import load_checkpoint
from training.metrics import (
    collect_interpretability_data,
    collect_predictions,
    collect_predictions_tta,
    compute_metrics,
)
from training.plots import (
    save_error_analysis_plots,
    save_model_interpretability_plots,
    save_quality_plots,
    save_split_distribution_plots,
)


for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HMST Architecture 1")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--mode", type=str, choices=["test", "full"], default="full",
                        help="Config mode to use for dataset settings")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["uwiqa", "saud", "uid"])
    args = parser.parse_args()

    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    config = load_config(args.mode)
    if args.dataset:
        config["dataset"] = args.dataset

    mode    = config.get("mode", args.mode or "full")
    dataset = config.get("dataset", "uwiqa")
    out_dir = dataset_output_dir(dataset, mode, project_root)

    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(config)

    print(f"\nEvaluating HMST Architecture 1")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print(f"Dataset:    {dataset}")
    print(f"Output dir: {out_dir}")

    # Dataset
    train_ds, val_ds, test_ds = build_datasets(config, project_root)
    split_plot_paths = save_split_distribution_plots(
        train_ds,
        val_ds,
        test_ds,
        output_dir=os.path.join(out_dir, "plots"),
        mos_range=(0.2, 1.0) if dataset == "uwiqa" else (0.0, 1.0),
    )
    print("Split distribution plots saved:")
    for plot_path in split_plot_paths:
        print(f"  {plot_path}")
    batch_size = config.get("batch_size", 4)
    num_workers = config.get("num_workers", 0)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    # Model
    patch_sizes  = config.get("patch_sizes", [32, 64, 128])
    image_size  = config.get("image_size", 224)

    # Read the checkpoint's stored config so tuned checkpoints load correctly,
    # and hard-fail if the checkpoint belongs to a different dataset.
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg  = ckpt_meta.get("config", {})
    ckpt_ds   = ckpt_cfg.get("dataset")
    if ckpt_ds is not None and ckpt_ds != dataset:
        raise RuntimeError(
            f"Checkpoint '{args.checkpoint}' records dataset '{ckpt_ds}' "
            f"but evaluation is configured for dataset '{dataset}'."
        )

    model = HMST(
        image_size=ckpt_cfg.get("image_size", image_size),
        patch_sizes=ckpt_cfg.get("patch_sizes", patch_sizes),
        d_model=int(ckpt_cfg.get("d_model", 512)),
        nhead=int(ckpt_cfg.get("nhead", 8)),
        nlayers=int(ckpt_cfg.get("nlayers", 2)),
        n_regions=int(ckpt_cfg.get("n_regions", 4)),
        dropout=float(ckpt_cfg.get("dropout", 0.3)),
    ).to(device)

    # Load checkpoint weights
    ckpt = load_checkpoint(args.checkpoint, model, device=device)
    trained_epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint from epoch {trained_epoch}")

    os.makedirs(out_dir, exist_ok=True)

    # Evaluate on val and test
    plot_results = {}
    for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
        # Standard single-pass evaluation
        result = collect_predictions(model, loader, device)
        metrics = compute_metrics(result["predictions"], result["targets"])

        # TTA evaluation (4-view flip average)
        result_tta = collect_predictions_tta(model, loader, device)
        metrics_tta = compute_metrics(result_tta["predictions"], result_tta["targets"])
        plot_results[split_name.lower()] = (result, result_tta, metrics_tta)

        # Per-image absolute errors (TTA + no-TTA) -> MAE / MedAE
        actuals = result_tta["targets"]
        preds_tta = result_tta["predictions"]
        preds_no_tta = result["predictions"]
        abs_errors_tta = np.abs(preds_tta - actuals)
        abs_errors_no_tta = np.abs(preds_no_tta - actuals)
        mae_tta = float(np.mean(abs_errors_tta))
        medae_tta = float(np.median(abs_errors_tta))
        mae_no_tta = float(np.mean(abs_errors_no_tta))
        medae_no_tta = float(np.median(abs_errors_no_tta))

        # Persist MAE / MedAE into the metrics dict (both TTA and no-TTA)
        metrics_tta.update({"mae": mae_tta, "medae": medae_tta})
        metrics.update({"mae": mae_no_tta, "medae": medae_no_tta})

        print(f"\n{split_name} Metrics (n={len(result['predictions'])}):")
        print(f"  {'Metric':<8}  {'No TTA':>8}  {'TTA (4-view)':>12}")
        print(f"  {'-'*34}")
        print(f"  {'PLCC':<8}  {metrics['plcc']:>8.4f}  {metrics_tta['plcc']:>12.4f}")
        print(f"  {'SROCC':<8}  {metrics['srocc']:>8.4f}  {metrics_tta['srocc']:>12.4f}")
        print(f"  {'KRCC':<8}  {metrics['krcc']:>8.4f}  {metrics_tta['krcc']:>12.4f}")
        print(f"  {'RMSE':<8}  {metrics['rmse']:>8.4f}  {metrics_tta['rmse']:>12.4f}")
        print(f"  {'MAE':<8}  {mae_no_tta:>8.4f}  {mae_tta:>12.4f}")
        print(f"  {'MedAE':<8}  {medae_no_tta:>8.4f}  {medae_tta:>12.4f}")

        # Save metrics JSON (TTA as primary, plus no-TTA reference)
        with open(os.path.join(out_dir, f"{split_name.lower()}_metrics.json"), "w") as f:
            json.dump(metrics_tta, f, indent=2)
        with open(os.path.join(out_dir, f"{split_name.lower()}_metrics_no_tta.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Per-image predictions CSV (Actual vs Predicted analysis)
        paths = result_tta.get("paths", result.get("paths", []))
        pred_csv_path = os.path.join(out_dir, f"{split_name.lower()}_predictions.csv")
        rows = []
        for idx in range(len(actuals)):
            full_path = paths[idx] if idx < len(paths) else f"{split_name.lower()}_{idx:04d}"
            rows.append({
                "image": os.path.basename(full_path),
                "actual_mos": float(actuals[idx]),
                "pred_mos_tta": float(preds_tta[idx]),
                "pred_mos_no_tta": float(preds_no_tta[idx]),
                "abs_error_tta": float(abs_errors_tta[idx]),
                "error_tta": float(preds_tta[idx] - actuals[idx]),
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

        print(f"  Metrics saved to:      {os.path.join(out_dir, f'{split_name.lower()}_metrics.json')}")
        print(f"  Per-image CSV saved to: {pred_csv_path}")

    if dataset in {"uwiqa", "saud", "uid"}:
        validation_result, validation_result_tta, validation_metrics = plot_results["validation"]
        test_result, test_result_tta, test_metrics = plot_results["test"]
        plot_paths = save_quality_plots(
            actual_test=test_result_tta["targets"],
            predicted_test_tta=test_result_tta["predictions"],
            test_metrics=test_metrics,
            actual_validation=validation_result_tta["targets"],
            predicted_validation_tta=validation_result_tta["predictions"],
            validation_metrics=validation_metrics,
            output_dir=os.path.join(out_dir, "plots"),
        )
        pareto_dataset = build_full_evaluation_dataset(config, project_root)
        pareto_result = test_result_tta
        if pareto_dataset is not None:
            pareto_loader = DataLoader(
                pareto_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            pareto_result = collect_predictions_tta(model, pareto_loader, device)
        error_plot_paths = save_error_analysis_plots(
            actual=test_result_tta["targets"],
            predicted_tta=test_result_tta["predictions"],
            predicted_no_tta=test_result["predictions"],
            image_paths=test_result_tta.get("paths", []),
            output_dir=os.path.join(out_dir, "plots"),
            pareto_actual=pareto_result["targets"],
            pareto_predicted=pareto_result["predictions"],
        )
        interpretability_data = collect_interpretability_data(model, test_loader, device)
        correspondence_matrix = build_correspondence_matrix(
            num_patches_fine=model.num_patches[0],
            num_patches_coarse=model.num_patches[2],
            grid_fine=model.grid_sizes[0],
            grid_coarse=model.grid_sizes[2],
        )
        interpretability_plot_paths = save_model_interpretability_plots(
            interpretability_data,
            model,
            correspondence_matrix,
            output_dir=os.path.join(out_dir, "plots"),
        )
        print("Correlation plots saved:")
        for plot_path in (*plot_paths, *error_plot_paths, *interpretability_plot_paths):
            print(f"  {plot_path}")


if __name__ == "__main__":
    main()
