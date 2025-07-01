#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3a_validate_knn.py
------------------
Validates a trained GNN model (FlowNet or Gao-RotFlowNet) on CFD validation datasets
using k-Nearest Neighbors (k-NN) graph construction.

Pipeline:
 1. Loads configuration and trained model checkpoint.
 2. Iterates through specified validation cases (e.g., sUbend_011, sUbend_012).
 3. For each VTK frame in a case:
    a. Constructs a k-NN graph from the VTK data (no down-sampling by default for validation).
    b. Performs inference with the loaded model to predict the velocity field.
    c. Computes metrics: MSE, RMSE of velocity magnitude, MSE of divergence.
    d. Saves the predicted velocity field and normalized pressure (if input 'p' exists) to a new VTK file.
 4. Aggregates and reports overall validation metrics.
 5. Optionally logs metrics and output files to W&B.

Usage:
    python scripts/3a_validate_knn.py --config path/to/config.yaml \
        --model-checkpoint path/to/your_model.pth \
        --model-name FlowNet \
        --val-data-dir data/CFD_Ubend_other_val \
        --output-dir outputs/my_run_validation_knn

If --config is provided, its values are used as defaults, overridden by CLI args.
"""

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
import meshio # For reading pressure from original file

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cfd_gnn.data_utils import vtk_to_knn_graph, sort_frames_by_number
from src.cfd_gnn.models import FlowNet, RotFlowNet # Or a generic loader
from src.cfd_gnn.losses import calculate_divergence
from src.cfd_gnn.metrics import cosine_similarity_metric # Add more if needed
from src.cfd_gnn.utils import (
    load_config, get_run_name, initialize_wandb, get_device,
    write_vtk_with_fields
)
import torch.nn.functional as F # For direct MSE


def main():
    parser = argparse.ArgumentParser(description="Validate a trained GNN model using k-NN graphs.")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "--model-checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--model-name", type=str, required=True, choices=["FlowNet", "Gao", "RotFlowNet"],
        help="Name of the model architecture corresponding to the checkpoint."
    )
    parser.add_argument(
        "--val-data-dir", type=str, help="Directory containing validation CFD cases (e.g., CFD_Ubend_other_val)."
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save predicted VTK files and metrics."
    )
    parser.add_argument(
        "--k-neighbors", type=int, help="Number of neighbors for k-NN graph construction."
    )
    parser.add_argument(
        "--no-downsample", action="store_true", help="Do not downsample points for k-NN graphs (recommended for validation)."
    )
    parser.add_argument(
        "--cases", type=str, nargs="*", help="Specific cases to validate (e.g., sUbend_011 sUbend_012). Default: all sUbend* cases."
    )
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu", "auto"], help="Override device from config."
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional run name for W&B logging context."
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging for this validation run."
    )

    args = parser.parse_args()

    # --- 1. Configuration Loading ---
    default_cfg_path = project_root / "config" / "default_config.yaml"
    cfg = load_config(args.config or default_cfg_path)

    # Override config with CLI arguments
    val_data_dir = Path(args.val_data_dir or cfg["val_root"])
    output_dir = Path(args.output_dir or project_root / cfg.get("output_base_dir", "outputs") / f"{args.model_name}_knn_validation_{get_run_name(None)}")

    k_neighbors = args.k_neighbors if args.k_neighbors is not None else cfg.get("graph_config",{}).get("k", 12)
    downsample_n_val = None if args.no_downsample else cfg.get("graph_config",{}).get("down_n") # Typically None for validation

    device = get_device(args.device or cfg.get("device", "auto"))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validation outputs will be saved to: {output_dir}")

    # --- W&B Initialization (Optional) ---
    val_run_name = get_run_name(args.run_name or f"val_knn_{args.model_name}_{val_data_dir.name}")
    if args.no_wandb:
        wandb_run = None
    else:
        wandb_run = initialize_wandb(
            config={**cfg, **vars(args)}, # Log combined config
            run_name=val_run_name,
            project_name=cfg.get("wandb_project") + "_Validation" if cfg.get("wandb_project") else "CFD_GNN_Validation",
            output_dir=output_dir # Place W&B logs within the validation output dir
        )

    # --- 2. Load Trained Model ---
    model_checkpoint_path = Path(args.model_checkpoint)
    if not model_checkpoint_path.exists():
        print(f"Error: Model checkpoint not found at {model_checkpoint_path}")
        sys.exit(1)

    models_registry = {"FlowNet": FlowNet, "Gao": RotFlowNet, "RotFlowNet": RotFlowNet}
    model_class = models_registry.get(args.model_name)
    if not model_class:
        print(f"Error: Unknown model name '{args.model_name}'. Supported: {list(models_registry.keys())}")
        sys.exit(1)

    # Model architecture config should ideally be part of checkpoint or known.
    # For now, use general config, assuming it matches the saved model.
    model_arch_cfg = {**cfg, **cfg.get(f"{args.model_name}_config", {})}
    model = model_class(model_arch_cfg).to(device)

    try:
        model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading model state_dict from {model_checkpoint_path}: {e}")
        print("Ensure model_name and config match the checkpoint.")
        sys.exit(1)

    model.eval()
    print(f"Loaded {args.model_name} model from: {model_checkpoint_path}")
    print(f"Using device: {device}")

    # --- 3. Validation Loop ---
    # Initialize structures for detailed metrics
    metrics_per_case_then_frame = {
        "mse": {}, "rmse_mag": {}, "mse_div": {}, "cosine_sim": {},
        "mse_x": {}, "mse_y": {}, "mse_z": {},
        "max_true_vel_mag": {}, "max_pred_vel_mag": {} # Added max velocity
    }
    all_frames_metrics_flat = {key: [] for key in metrics_per_case_then_frame}

    # Setup CSV for per-frame metrics
    frame_metrics_csv_path = output_dir / f"frame_metrics_knn_{args.model_name}.csv"
    detailed_csv_file = open(frame_metrics_csv_path, "w", newline="")
    detailed_csv_writer = csv.writer(detailed_csv_file)
    detailed_csv_header = [
        "case_name", "frame_file", "mse", "rmse_mag", "mse_div", "cosine_sim",
        "mse_x", "mse_y", "mse_z", "max_true_vel_mag", "max_pred_vel_mag"
    ]
    detailed_csv_writer.writerow(detailed_csv_header)

    # Setup CSV for per-frame-slice metrics
    slice_metrics_csv_path = None
    slice_csv_writer = None
    if cfg.get("slice_analysis", {}).get("enabled", False):
        slice_metrics_csv_path = output_dir / f"frame_slice_metrics_knn_{args.model_name}.csv"
        slice_csv_file = open(slice_metrics_csv_path, "w", newline="")
        slice_csv_writer = csv.writer(slice_csv_file)
        slice_csv_header = [
            "case_name", "frame_file", "slice_axis", "slice_idx_in_axis",
            "slice_position", "num_points_in_slice",
            "max_true_vel_mag", "avg_true_vel_mag", "max_pred_vel_mag", "avg_pred_vel_mag"
        ]
        slice_csv_writer.writerow(slice_csv_header)
        print(f"Logging detailed per-frame-slice metrics to: {slice_metrics_csv_path}")
        all_slice_metrics_aggregated = {}
    print(f"Logging detailed per-frame metrics to: {frame_metrics_csv_path}")

    # Determine cases to process
    if args.cases:
        case_names_to_process = args.cases
    elif cfg.get("cases_to_process"):
        case_names_to_process = cfg["cases_to_process"]
    else: # Default: find all sUbend* directories
        case_names_to_process = [d.name for d in val_data_dir.iterdir() if d.is_dir() and d.name.startswith("sUbend")]

    if not case_names_to_process:
        print(f"Error: No validation cases found or specified in {val_data_dir}.")
        sys.exit(1)
    print(f"Validating on cases: {case_names_to_process}")

    # Graph construction config for validation
    graph_cfg_val = {
        "k": k_neighbors,
        "down_n": downsample_n_val,
        "velocity_key": cfg.get("velocity_key", "U"),
        # For validation, we usually use original (non-noisy) data as input to model,
        # unless the model was specifically trained to denoise *and* predict.
        # Assuming `use_noisy_data=False` for kNN graph creation here.
        "use_noisy_data": False,
        "noisy_velocity_key_suffix": cfg.get("noisy_velocity_key_suffix", "_noisy")
    }

    for case_name in case_names_to_process:
        # Initialize lists for this case's frame metrics
        for metric_key in metrics_per_case_then_frame:
            metrics_per_case_then_frame[metric_key][case_name] = []

        case_data_dir = val_data_dir / case_name / "CFD"
        case_output_dir = output_dir / args.model_name / case_name / "CFD"
        case_output_dir.mkdir(parents=True, exist_ok=True)

        if not case_data_dir.is_dir():
            print(f"Warning: Data directory for case {case_name} not found at {case_data_dir}. Skipping.")
            continue

        vtk_files = sort_frames_by_number(list(case_data_dir.glob("*.vtk")))
        if not vtk_files:
            print(f"Warning: No VTK files found in {case_data_dir} for case {case_name}. Skipping.")
            continue

        print(f"\nProcessing Case: {case_name} ({len(vtk_files)} frames)")

        for frame_idx, vtk_path in enumerate(vtk_files):
            print(f"  Frame: {vtk_path.name} ({frame_idx+1}/{len(vtk_files)})")

            # a. Construct k-NN graph
            try:
                # We need graph_t0 for input, and graph_t1 (next frame) for target if comparing predictions over time.
                # For single frame validation (predicting current frame's velocity from its own geometry),
                # we'd need a different setup or assume model predicts U from U.
                # The original script's logic seems to imply predicting U(t+1) from U(t).
                # For this validation script, let's assume the task is to "process" the current frame.
                # If the model predicts velocity for the *same* frame (e.g., denoising or super-resolution),
                # then `true_vel` comes from `graph.x`.
                # If model predicts *next* frame, this script needs pairs.
                # `tranv3_val_cpu.py` implies processing current frame: `real = graph.x`

                graph = vtk_to_knn_graph(vtk_path, **graph_cfg_val, device=device)
                true_vel = graph.x.to(device) # Ground truth velocity from the input VTK
                points_np = graph.pos.cpu().numpy()

                # Read original pressure for saving (if exists)
                original_mesh = meshio.read(str(vtk_path))
                pressure_np = None
                if cfg.get("pressure_key", "p") in original_mesh.point_data:
                    p_all = original_mesh.point_data[cfg["pressure_key"]].astype(np.float32)
                    # Normalize pressure (same logic as full_mesh_validation)
                    inlet_idx = int(np.argmin(points_np[:, 1])) # Smallest y-coord
                    p_ref = float(p_all[inlet_idx])
                    diffp = p_all - p_ref
                    normP = float(np.linalg.norm(diffp))
                    if normP == 0.0: normP = 1.0
                    pressure_np = (p_all - p_ref) / normP
                else:
                    print(f"    Warning: Pressure key '{cfg.get('pressure_key','p')}' not found in {vtk_path}. Pressure field will not be saved.")

            except Exception as e:
                print(f"    Error processing VTK or building graph for {vtk_path.name}: {e}")
                continue

            # b. Perform inference
            with torch.no_grad():
                predicted_vel = model(graph).cpu() # Move to CPU for metrics and saving

            # c. Compute metrics
            mse = F.mse_loss(predicted_vel, true_vel.cpu()).item()

            pred_mag = predicted_vel.norm(dim=1)
            true_mag = true_vel.cpu().norm(dim=1)
            rmse_mag = torch.sqrt(F.mse_loss(pred_mag, true_mag)).item()

            # Divergence of predicted field (target is implicitly zero)
            # Note: calculate_divergence needs graph on same device as velocity
            div_pred_torch = calculate_divergence(predicted_vel.to(device), graph.to(device))
            mse_div = (div_pred_torch.cpu() ** 2).mean().item()

            cos_sim = cosine_similarity_metric(predicted_vel, true_vel.cpu())

            # Component-wise MSE
            mse_x = F.mse_loss(predicted_vel[:, 0], true_vel.cpu()[:, 0]).item()
            mse_y = F.mse_loss(predicted_vel[:, 1], true_vel.cpu()[:, 1]).item()
            mse_z = F.mse_loss(predicted_vel[:, 2], true_vel.cpu()[:, 2]).item()

            # Max velocity magnitudes
            max_true_vel_mag = true_vel.cpu().norm(dim=1).max().item()
            max_pred_vel_mag = predicted_vel.norm(dim=1).max().item()

            # Store metrics for this frame (for this case)
            metrics_per_case_then_frame["mse"][case_name].append(mse)
            metrics_per_case_then_frame["rmse_mag"][case_name].append(rmse_mag)
            metrics_per_case_then_frame["mse_div"][case_name].append(mse_div)
            metrics_per_case_then_frame["cosine_sim"][case_name].append(cos_sim)
            metrics_per_case_then_frame["mse_x"][case_name].append(mse_x)
            metrics_per_case_then_frame["mse_y"][case_name].append(mse_y)
            metrics_per_case_then_frame["mse_z"][case_name].append(mse_z)
            metrics_per_case_then_frame["max_true_vel_mag"][case_name].append(max_true_vel_mag)
            metrics_per_case_then_frame["max_pred_vel_mag"][case_name].append(max_pred_vel_mag)

            # Store for global flat list
            all_frames_metrics_flat["mse"].append(mse)
            all_frames_metrics_flat["rmse_mag"].append(rmse_mag)
            all_frames_metrics_flat["mse_div"].append(mse_div)
            all_frames_metrics_flat["cosine_sim"].append(cos_sim)
            all_frames_metrics_flat["mse_x"].append(mse_x)
            all_frames_metrics_flat["mse_y"].append(mse_y)
            all_frames_metrics_flat["mse_z"].append(mse_z)
            all_frames_metrics_flat["max_true_vel_mag"].append(max_true_vel_mag)
            all_frames_metrics_flat["max_pred_vel_mag"].append(max_pred_vel_mag)

            print(f"    Metrics: MSE={mse:.4e}, RMSE_mag={rmse_mag:.4e}, MSE_div={mse_div:.4e}, CosSim={cos_sim:.4f}")

            # Write to detailed CSV
            detailed_csv_writer.writerow([
                case_name, vtk_path.name, mse, rmse_mag, mse_div, cos_sim,
                mse_x, mse_y, mse_z, max_true_vel_mag, max_pred_vel_mag
            ])

            # Calculate and log slice metrics if enabled
            if slice_csv_writer: # Implies slice_analysis is enabled
                from src.cfd_gnn.metrics import calculate_slice_metrics_for_frame # Lazy import

                slice_metrics_current_frame = calculate_slice_metrics_for_frame(
                    points_np=graph.pos.cpu().numpy(),
                    true_velocity_np=true_vel.cpu().numpy(),
                    pred_velocity_np=predicted_vel.cpu().numpy(),
                    slice_config=cfg.get("slice_analysis")
                )
                for key, value in slice_metrics_current_frame.items():
                    parts = key.split('_')
                    if len(parts) >= 5 and parts[0] == "slice":
                        s_axis, s_idx, s_pos_str = parts[1], parts[2], parts[3].replace("pos","")
                        s_metric_name = "_".join(parts[4:])

                        if s_metric_name == "max_true_vel_mag":
                            slice_csv_writer.writerow([
                                case_name, vtk_path.name, s_axis, s_idx, float(s_pos_str),
                                slice_metrics_current_frame.get(f"slice_{s_axis}_{s_idx}_pos{float(s_pos_str):.2f}_num_points", 0),
                                slice_metrics_current_frame.get(f"slice_{s_axis}_{s_idx}_pos{float(s_pos_str):.2f}_max_true_vel_mag", np.nan),
                                slice_metrics_current_frame.get(f"slice_{s_axis}_{s_idx}_pos{float(s_pos_str):.2f}_avg_true_vel_mag", np.nan),
                                slice_metrics_current_frame.get(f"slice_{s_axis}_{s_idx}_pos{float(s_pos_str):.2f}_max_pred_vel_mag", np.nan),
                                slice_metrics_current_frame.get(f"slice_{s_axis}_{s_idx}_pos{float(s_pos_str):.2f}_avg_pred_vel_mag", np.nan),
                            ])

                        agg_key = f"slice_{s_axis}_{s_idx}_{s_metric_name}"
                        if not np.isnan(value):
                            all_slice_metrics_aggregated.setdefault(agg_key, []).append(value)

            # d. Save predicted VTK
            frame_base_name = vtk_path.stem # e.g., "Frame_00_data"
            output_vtk_path = case_output_dir / f"{frame_base_name}_pred_{args.model_name}_knn.vtk"

            point_data_to_save = {cfg.get("predicted_velocity_key", "velocity"): predicted_vel.numpy()}
            if pressure_np is not None:
                point_data_to_save[cfg.get("predicted_pressure_key", "pressure")] = pressure_np

            write_vtk_with_fields(
                output_vtk_path,
                points=points_np,
                point_data=point_data_to_save
            )
            # print(f"    Saved prediction to: {output_vtk_path}")

    if slice_csv_writer:
        slice_csv_file.close()
    detailed_csv_file.close()

    # --- 4. Aggregate and Report Metrics ---
    # 1. Overall metrics (mean and std over all frames from all cases)
    summary_overall_metrics = {}
    print(f"\n--- Overall Validation Summary (k-NN, {args.model_name}, All Cases & Frames) ---")
    for metric_name, flat_values_list in all_frames_metrics_flat.items():
        if flat_values_list:
            mean_val = float(np.mean(flat_values_list))
            std_val = float(np.std(flat_values_list))
            summary_overall_metrics[f"Overall_Val_KNN/{args.model_name}/{metric_name}_mean"] = mean_val
            summary_overall_metrics[f"Overall_Val_KNN/{args.model_name}/{metric_name}_std"] = std_val
            print(f"  Avg {metric_name}: {mean_val:.4e} ± {std_val:.4e} (n={len(flat_values_list)})")
            if wandb_run:
                wandb_run.log({f"Histograms_All_Frames_KNN/{args.model_name}/{metric_name}": wandb.Histogram(np.array(flat_values_list))})

    # 2. Per-case summary metrics
    summary_per_case_metrics = {}
    print(f"\n  Per-Case Summary Metrics (k-NN, {args.model_name}):")
    for case_name_processed in case_names_to_process:
        print(f"    Case: {case_name_processed}")
        for metric_name, case_metric_data in metrics_per_case_then_frame.items():
            values_for_case = case_metric_data.get(case_name_processed, [])
            if values_for_case:
                mean_val = float(np.mean(values_for_case))
                std_val = float(np.std(values_for_case))
                summary_per_case_metrics[f"PerCase_Val_KNN/{args.model_name}/{case_name_processed}/{metric_name}_mean"] = mean_val
                summary_per_case_metrics[f"PerCase_Val_KNN/{args.model_name}/{case_name_processed}/{metric_name}_std"] = std_val
                print(f"      Avg {metric_name}: {mean_val:.4e} ± {std_val:.4e} (n={len(values_for_case)})")

    # 3. Aggregated Slice Metrics for W&B
    summary_slice_metrics_wandb = {}
    if cfg.get("slice_analysis", {}).get("enabled", False) and all_slice_metrics_aggregated:
        print(f"\n  Aggregated Slice Metrics (k-NN, {args.model_name}):")
        for agg_key, values_list in all_slice_metrics_aggregated.items():
            if values_list:
                mean_val = float(np.mean(values_list))
                std_val = float(np.std(values_list))
                summary_slice_metrics_wandb[f"SliceMetrics_KNN/{args.model_name}/{agg_key}_mean"] = mean_val
                summary_slice_metrics_wandb[f"SliceMetrics_KNN/{args.model_name}/{agg_key}_std"] = std_val
                print(f"    {agg_key}: Mean={mean_val:.4e}, Std={std_val:.4e} (n={len(values_list)})")

    # Save overall summary metrics to a CSV file (consistent with combined_validation)
    summary_csv_path = output_dir / f"validation_summary_knn_{args.model_name}.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_group", "model", "metric_name", "value_type", "value"])
        # Log overall metrics
        for key, val in summary_overall_metrics.items():
            parts = key.split('/') # e.g., "Overall_Val_KNN/FlowNet/mse_mean"
            writer.writerow([parts[0], parts[1], parts[2].replace('_mean','').replace('_std',''), parts[2].split('_')[-1], val])
        # Optionally, log per-case metrics to the same CSV or a different one
        # For now, keeping main summary CSV for overall metrics only to match combined_validation script.

    print(f"Saved overall summary metrics to: {summary_csv_path}")

    if wandb_run:
        if summary_overall_metrics:
            wandb_run.log(summary_overall_metrics)
        if summary_per_case_metrics:
            wandb_run.log(summary_per_case_metrics)
        if summary_slice_metrics_wandb:
            wandb_run.log(summary_slice_metrics_wandb)

        # Log the detailed frame metrics CSV as an artifact
        if frame_metrics_csv_path.exists(): # Check if file was created
            art_detailed_metrics = wandb.Artifact(f"frame_metrics_knn_{args.model_name}", type="per_frame_metrics")
            art_detailed_metrics.add_file(str(frame_metrics_csv_path))
            wandb_run.log_artifact(art_detailed_metrics)

        if slice_metrics_csv_path and slice_metrics_csv_path.exists():
            art_slice_metrics = wandb.Artifact(f"frame_slice_metrics_knn_{args.model_name}", type="per_frame_slice_metrics")
            art_slice_metrics.add_file(str(slice_metrics_csv_path))
            wandb_run.log_artifact(art_slice_metrics)

        wandb_run.finish()

    print("\nk-NN validation script finished.")

if __name__ == "__main__":
    main()
