#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_combined_validation.py
------------------------
Orchestrates a comprehensive validation pipeline for a trained GNN model.
It combines model inference with basic metric calculation (like scripts 3a or 3b)
and subsequent histogram-based JSD validation (like script 4).

Pipeline:
 1. Loads configuration, trained model checkpoint.
 2. Part 1: Model Inference & Basic Metrics
    a. Iterates through validation cases.
    b. Constructs graphs (k-NN or full_mesh based on user choice).
    c. Performs model inference to get velocity predictions.
    d. Computes and logs basic metrics (MSE, RMSE_mag, MSE_div, CosSim).
    e. Saves predicted VTK files (velocity + normalized pressure) to a dedicated
       subdirectory (e.g., .../combined_run_outputs/model_predictions/FlowNet/).
 3. Part 2: Histogram JSD Validation
    a. Uses the original validation data as the "real" reference for JSD.
    b. Uses the predictions saved in Part 1 as the "predicted" data for JSD.
    c. Computes JSD per point and saves heatmaps to another subdirectory
       (e.g., .../combined_run_outputs/jsd_heatmaps/FlowNet/).
    d. Logs JSD metrics.
 4. Logs all metrics and relevant artifacts to W&B if enabled.

Usage:
    python scripts/5_combined_validation.py --config path/to/config.yaml \
        --model-checkpoint path/to/model.pth --model-name FlowNet \
        --val-data-dir data/CFD_Ubend_other_val \
        --output-dir outputs/my_combined_validation_run \
        --graph-type knn [or full_mesh]

If --config is provided, its values are used as defaults, overridden by CLI args.
"""

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
import shutil

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cfd_gnn.data_utils import vtk_to_knn_graph, vtk_to_fullmesh_graph, sort_frames_by_number
from src.cfd_gnn.models import FlowNet, RotFlowNet
from src.cfd_gnn.losses import calculate_divergence
from src.cfd_gnn.metrics import cosine_similarity_metric
from src.cfd_gnn.validation import histogram_jsd_validation # Reusing this directly
from src.cfd_gnn.utils import (
    load_config, get_run_name, initialize_wandb, get_device,
    write_vtk_with_fields
)
import torch.nn.functional as F # For direct MSE

def run_inference_and_basic_metrics(
    model: torch.nn.Module,
    val_data_dir: Path,
    output_pred_dir: Path, # Base directory for this model's predictions
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    wandb_run: 'wandb.sdk.wandb_run.Run | None'
) -> dict:
    """
    Performs model inference, computes basic metrics, and saves prediction VTKs.
    Returns a dictionary of aggregated basic metrics.
    """
    print(f"\n--- Part 1: Inference & Basic Metrics ({args.model_name}, Graph: {args.graph_type}) ---")
    model.eval()
    all_basic_metrics = {"mse": [], "rmse_mag": [], "mse_div": [], "cosine_sim": []}

    if args.cases:
        case_names_to_process = args.cases
    elif cfg.get("cases_to_process"):
        case_names_to_process = cfg["cases_to_process"]
    else:
        case_names_to_process = [d.name for d in val_data_dir.iterdir() if d.is_dir() and d.name.startswith("sUbend")]

    if not case_names_to_process:
        print(f"Warning: No cases found in {val_data_dir} for inference. Skipping Part 1.")
        return {}
    print(f"Processing cases for inference: {case_names_to_process}")

    # Graph construction config
    if args.graph_type == "knn":
        graph_cfg_inference = {
            "k": args.k_neighbors if args.k_neighbors is not None else cfg.get("graph_config",{}).get("k", 12),
            "down_n": None if args.no_downsample_knn else cfg.get("graph_config",{}).get("down_n"),
            "velocity_key": cfg.get("velocity_key", "U"),
            "use_noisy_data": False, # Typically validate on clean geometry
            "noisy_velocity_key_suffix": cfg.get("noisy_velocity_key_suffix", "_noisy")
        }
    else: # full_mesh
        graph_cfg_inference = {
            "velocity_key": cfg.get("velocity_key", "U"),
            "pressure_key": cfg.get("pressure_key", "p")
        }

    for case_name in case_names_to_process:
        case_val_cfd_dir = val_data_dir / case_name / "CFD"
        # Output for this specific case's predictions
        case_pred_output_cfd_dir = output_pred_dir / args.model_name / case_name / "CFD"
        case_pred_output_cfd_dir.mkdir(parents=True, exist_ok=True)

        if not case_val_cfd_dir.is_dir():
            print(f"Warning: Data for case {case_name} not found at {case_val_cfd_dir}. Skipping.")
            continue

        vtk_files = sort_frames_by_number(list(case_val_cfd_dir.glob("*.vtk")))
        if not vtk_files:
            print(f"Warning: No VTK files in {case_val_cfd_dir}. Skipping case {case_name}.")
            continue

        print(f"  Case: {case_name} ({len(vtk_files)} frames)")
        for vtk_path in vtk_files:
            try:
                if args.graph_type == "knn":
                    graph = vtk_to_knn_graph(vtk_path, **graph_cfg_inference, device=device)
                else: # full_mesh
                    graph = vtk_to_fullmesh_graph(vtk_path, **graph_cfg_inference, device=device)

                true_vel = graph.x.to(device)
                points_np = graph.pos.cpu().numpy()
                pressure_np = graph.p_norm.cpu().numpy() if hasattr(graph, 'p_norm') else None

                if pressure_np is None and args.graph_type == "knn": # Need to load pressure for KNN if saving
                    import meshio # Lazy import
                    original_mesh = meshio.read(str(vtk_path))
                    if cfg.get("pressure_key", "p") in original_mesh.point_data:
                        p_all = original_mesh.point_data[cfg["pressure_key"]].astype(np.float32)
                        inlet_idx = int(np.argmin(points_np[:, 1]))
                        p_ref = float(p_all[inlet_idx])
                        diffp = p_all - p_ref
                        normP = float(np.linalg.norm(diffp)); normP = normP if normP > 1e-9 else 1.0
                        pressure_np = (p_all - p_ref) / normP


                with torch.no_grad():
                    predicted_vel = model(graph).cpu()

                mse = F.mse_loss(predicted_vel, true_vel.cpu()).item()
                rmse_mag = torch.sqrt(F.mse_loss(predicted_vel.norm(dim=1), true_vel.cpu().norm(dim=1))).item()
                div_pred_torch = calculate_divergence(predicted_vel.to(device), graph.to(device))
                mse_div = (div_pred_torch.cpu() ** 2).mean().item()
                cos_sim = cosine_similarity_metric(predicted_vel, true_vel.cpu())

                all_basic_metrics["mse"].append(mse)
                all_basic_metrics["rmse_mag"].append(rmse_mag)
                all_basic_metrics["mse_div"].append(mse_div)
                all_basic_metrics["cosine_sim"].append(cos_sim)

                # Save prediction VTK
                frame_base_name = vtk_path.stem
                output_vtk_path = case_pred_output_cfd_dir / f"{frame_base_name}_pred_{args.model_name}_{args.graph_type}.vtk"
                point_data_to_save = {cfg.get("predicted_velocity_key", "velocity"): predicted_vel.numpy()}
                if pressure_np is not None:
                    point_data_to_save[cfg.get("predicted_pressure_key", "pressure")] = pressure_np
                write_vtk_with_fields(output_vtk_path, points_np, point_data_to_save)

            except Exception as e:
                print(f"    Error during inference for {vtk_path.name} in case {case_name}: {e}")
                # import traceback; traceback.print_exc() # For debugging
                continue

    # Aggregate and log basic metrics
    summary_basic_metrics = {}
    print(f"\n  Summary of Basic Metrics ({args.model_name}, Graph: {args.graph_type}):")
    for name, values in all_basic_metrics.items():
        if values:
            mean_val, std_val = float(np.mean(values)), float(np.std(values))
            summary_basic_metrics[f"Basic/{args.model_name}/{name}_mean"] = mean_val
            summary_basic_metrics[f"Basic/{args.model_name}/{name}_std"] = std_val
            print(f"    Avg {name}: {mean_val:.4e} ± {std_val:.4e}")

    if wandb_run and summary_basic_metrics:
        wandb_run.log(summary_basic_metrics)

    return summary_basic_metrics


def main():
    parser = argparse.ArgumentParser(description="Run combined validation (Inference + JSD) for a GNN model.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to trained model .pth.")
    parser.add_argument("--model-name", type=str, required=True, choices=["FlowNet", "Gao", "RotFlowNet"], help="Model architecture.")
    parser.add_argument("--val-data-dir", type=str, help="Directory of original validation CFD cases.")
    parser.add_argument("--output-dir", type=str, help="Base directory for all outputs of this combined run.")
    parser.add_argument("--graph-type", type=str, choices=["knn", "full_mesh"], default="knn", help="Graph type for inference part.")

    # kNN specific args (ignored if graph-type is full_mesh)
    parser.add_argument("--k-neighbors", type=int, help="k for kNN graph construction.")
    parser.add_argument("--no-downsample-knn", action="store_true", help="Disable point downsampling for kNN.")

    parser.add_argument("--cases", type=str, nargs="*", help="Specific cases to validate.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], help="Device override.")
    parser.add_argument("--run-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B.")
    parser.add_argument("--skip-jsd", action="store_true", help="Skip the JSD histogram validation part.")

    args = parser.parse_args()

    # --- Configuration ---
    default_cfg_path = project_root / "config" / "default_config.yaml"
    cfg = load_config(args.config or default_cfg_path)

    val_data_dir = Path(args.val_data_dir or cfg["val_root"])
    # Main output directory for this combined validation run
    main_output_dir = Path(args.output_dir or project_root / cfg.get("output_base_dir", "outputs") / f"combined_val_{args.model_name}_{args.graph_type}_{get_run_name(None)}")
    main_output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device or cfg.get("device", "auto"))
    print(f"Combined Validation Run. Outputs: {main_output_dir}. Device: {device}.")

    # Subdirectory for predictions from Part 1
    output_dir_model_preds = main_output_dir / "model_predictions"
    output_dir_model_preds.mkdir(parents=True, exist_ok=True)

    # Subdirectory for JSD heatmaps from Part 2
    output_dir_jsd_heatmaps = main_output_dir / "jsd_heatmaps"
    # This will be created by histogram_jsd_validation if it runs

    # --- W&B ---
    combined_run_name = get_run_name(args.run_name or f"combo_{args.model_name}_{args.graph_type}_{val_data_dir.name}")
    if args.no_wandb:
        wandb_run = None
    else:
        wandb_run = initialize_wandb(
            config={**cfg, **vars(args)}, run_name=combined_run_name,
            project_name=cfg.get("wandb_project") + "_CombinedValidation" if cfg.get("wandb_project") else "CFD_GNN_CombinedVal",
            output_dir=main_output_dir
        )

    # --- Load Model ---
    model_checkpoint_path = Path(args.model_checkpoint)
    if not model_checkpoint_path.exists():
        print(f"Error: Model checkpoint {model_checkpoint_path} not found."); sys.exit(1)

    models_registry = {"FlowNet": FlowNet, "Gao": RotFlowNet, "RotFlowNet": RotFlowNet}
    model_class = models_registry[args.model_name]
    model_arch_cfg = {**cfg, **cfg.get(f"{args.model_name}_config", {})}
    model = model_class(model_arch_cfg).to(device)
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    print(f"Loaded {args.model_name} from {model_checkpoint_path}.")

    # --- Part 1: Inference and Basic Metrics ---
    basic_metrics_summary = run_inference_and_basic_metrics(
        model, val_data_dir, output_dir_model_preds, cfg, args, device, wandb_run
    )

    # Save basic metrics summary to CSV in main_output_dir
    summary_csv_path = main_output_dir / f"summary_basic_metrics_{args.model_name}_{args.graph_type}.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_group", "model", "metric_name", "value_type", "value"])
        for key, val in basic_metrics_summary.items():
            parts = key.split('/') # e.g., "Basic/FlowNet/mse_mean"
            writer.writerow([parts[0], parts[1], parts[2].replace('_mean','').replace('_std',''), parts[2].split('_')[-1], val])
    print(f"Saved basic metrics summary to: {summary_csv_path}")


    # --- Part 2: Histogram JSD Validation ---
    if not args.skip_jsd:
        print(f"\n--- Part 2: Histogram JSD Validation ({args.model_name}) ---")
        # Reference data for JSD is the original validation data
        jsd_ref_data_dir = val_data_dir
        # Predicted data for JSD comes from Part 1's outputs
        jsd_pred_data_dir = output_dir_model_preds / args.model_name # Path to specific model's predictions

        if not jsd_pred_data_dir.exists() or not any(jsd_pred_data_dir.iterdir()):
            print(f"Warning: Predicted data directory {jsd_pred_data_dir} for JSD is empty or missing. Skipping JSD validation.")
        else:
            jsd_run_config = {
                "reference_data_root": str(jsd_ref_data_dir),
                "predicted_data_root": str(jsd_pred_data_dir),
                "output_dir_jsd": str(output_dir_jsd_heatmaps), # Specific subdir for JSD VTKs
                "model_name_prefix": f"{args.model_name}_CombinedJSD", # For W&B logging and subdirs within output_dir_jsd_heatmaps
                "velocity_key_ref": cfg.get("velocity_key", "U"), # Key in original val data
                "velocity_key_pred": cfg.get("predicted_velocity_key", "velocity") # Key in VTKs saved by Part 1
            }
            # Use global config for nbins for JSD
            jsd_cfg = {**cfg, "nbins": cfg.get("nbins_jsd", cfg.get("nbins",32)), "BINS": cfg.get("nbins_jsd", cfg.get("nbins",32))}

            histogram_jsd_validation(
                model=None, # Not needed as we use pre-computed predictions
                cfg=jsd_cfg,
                run_config=jsd_run_config,
                device=device, # Not strictly used by hist_val when model is None, but good practice
                wandb_run=wandb_run # Pass the same W&B run object
            )
    else:
        print("\nSkipping Part 2: Histogram JSD Validation as per --skip-jsd.")

    if wandb_run:
        # Log final artifacts if any (e.g., the prediction directory)
        # This could be large, consider sampling or only logging summary files.
        # art_combined_outputs = wandb.Artifact(f"CombinedVal_{args.model_name}_{args.graph_type}", type="validation_run")
        # art_combined_outputs.add_file(str(summary_csv_path)) # Log summary of basic metrics
        # If JSD was run and produced files in output_dir_jsd_heatmaps, log that too.
        # if not args.skip_jsd and output_dir_jsd_heatmaps.exists() and any(output_dir_jsd_heatmaps.rglob('*.vtk')):
        #    art_combined_outputs.add_dir(str(output_dir_jsd_heatmaps), name="jsd_heatmaps")
        # if output_dir_model_preds.exists() and any(output_dir_model_preds.rglob('*.vtk')):
        #    art_combined_outputs.add_dir(str(output_dir_model_preds), name="model_predictions") # Potentially very large!
        # wandb_run.log_artifact(art_combined_outputs)
        wandb_run.finish()

    print("\nCombined validation script finished.")
    print(f"All outputs saved in: {main_output_dir}")

if __name__ == "__main__":
    main()
