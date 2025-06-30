#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3b_validate_full_mesh.py
------------------------
Validates a trained GNN model (FlowNet or Gao-RotFlowNet) on CFD validation datasets
using full mesh graph construction (from tetrahedral elements).

Pipeline:
 1. Loads configuration and trained model checkpoint.
 2. Iterates through specified validation cases (e.g., sUbend_011, sUbend_012).
 3. For each VTK frame in a case:
    a. Constructs a graph from the VTK's tetrahedral cell connectivity.
    b. Performs inference with the loaded model to predict the velocity field.
    c. Computes metrics: MSE, RMSE of velocity magnitude, MSE of divergence.
    d. Saves the predicted velocity field and normalized pressure to a new VTK file.
 4. Aggregates and reports overall validation metrics.
 5. Optionally logs metrics and output files to W&B.

Usage:
    python scripts/3b_validate_full_mesh.py --config path/to/config.yaml \
        --model-checkpoint path/to/your_model.pth \
        --model-name FlowNet \
        --val-data-dir data/CFD_Ubend_other_val \
        --output-dir outputs/my_run_validation_full_mesh

If --config is provided, its values are used as defaults, overridden by CLI args.
"""

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
# meshio is used by vtk_to_fullmesh_graph internally

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.cfd_gnn.data_utils import vtk_to_fullmesh_graph, sort_frames_by_number
from src.cfd_gnn.models import FlowNet, RotFlowNet # Or a generic loader
from src.cfd_gnn.losses import calculate_divergence
from src.cfd_gnn.metrics import cosine_similarity_metric # Add more if needed
from src.cfd_gnn.utils import (
    load_config, get_run_name, initialize_wandb, get_device,
    write_vtk_with_fields
)
import torch.nn.functional as F # For direct MSE


def main():
    parser = argparse.ArgumentParser(description="Validate a trained GNN model using full mesh graphs.")
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
    output_dir = Path(args.output_dir or project_root / cfg.get("output_base_dir", "outputs") / f"{args.model_name}_fullmesh_validation_{get_run_name(None)}")

    device = get_device(args.device or cfg.get("device", "auto"))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Validation outputs will be saved to: {output_dir}")

    # --- W&B Initialization (Optional) ---
    val_run_name = get_run_name(args.run_name or f"val_fullmesh_{args.model_name}_{val_data_dir.name}")
    if args.no_wandb:
        wandb_run = None
    else:
        wandb_run = initialize_wandb(
            config={**cfg, **vars(args)}, # Log combined config
            run_name=val_run_name,
            project_name=cfg.get("wandb_project") + "_Validation" if cfg.get("wandb_project") else "CFD_GNN_Validation",
            output_dir=output_dir
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
    all_metrics = {"mse": [], "rmse_mag": [], "mse_div": [], "cosine_sim": []}

    if args.cases:
        case_names_to_process = args.cases
    elif cfg.get("cases_to_process"):
        case_names_to_process = cfg["cases_to_process"]
    else:
        case_names_to_process = [d.name for d in val_data_dir.iterdir() if d.is_dir() and d.name.startswith("sUbend")]

    if not case_names_to_process:
        print(f"Error: No validation cases found or specified in {val_data_dir}.")
        sys.exit(1)
    print(f"Validating on cases: {case_names_to_process}")

    # Graph construction config keys for full_mesh
    velocity_key_graph = cfg.get("velocity_key", "U")
    pressure_key_graph = cfg.get("pressure_key", "p")


    for case_name in case_names_to_process:
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

            # a. Construct full mesh graph
            try:
                graph = vtk_to_fullmesh_graph(
                    vtk_path,
                    velocity_key=velocity_key_graph,
                    pressure_key=pressure_key_graph,
                    device=device
                )
                true_vel = graph.x.to(device) # Ground truth velocity from the input VTK
                points_np = graph.pos.cpu().numpy()
                # Normalized pressure is directly available from graph.p_norm
                pressure_np = graph.p_norm.cpu().numpy()

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

            div_pred_torch = calculate_divergence(predicted_vel.to(device), graph.to(device))
            mse_div = (div_pred_torch.cpu() ** 2).mean().item()

            cos_sim = cosine_similarity_metric(predicted_vel, true_vel.cpu())

            all_metrics["mse"].append(mse)
            all_metrics["rmse_mag"].append(rmse_mag)
            all_metrics["mse_div"].append(mse_div)
            all_metrics["cosine_sim"].append(cos_sim)

            print(f"    Metrics: MSE={mse:.4e}, RMSE_mag={rmse_mag:.4e}, MSE_div={mse_div:.4e}, CosSim={cos_sim:.4f}")

            # d. Save predicted VTK
            frame_base_name = vtk_path.stem
            output_vtk_path = case_output_dir / f"{frame_base_name}_pred_{args.model_name}_fullmesh.vtk"

            point_data_to_save = {
                cfg.get("predicted_velocity_key", "velocity"): predicted_vel.numpy(),
                cfg.get("predicted_pressure_key", "pressure"): pressure_np
            }

            write_vtk_with_fields(
                output_vtk_path,
                points=points_np,
                point_data=point_data_to_save
            )

    # --- 4. Aggregate and Report Metrics ---
    print("\n--- Overall Validation Summary (Full Mesh) ---")
    avg_metrics_summary = {}
    for metric_name, values in all_metrics.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            avg_metrics_summary[f"{metric_name}_mean"] = mean_val
            avg_metrics_summary[f"{metric_name}_std"] = std_val
            print(f"  Avg {metric_name.replace('_', ' '):<10}: {mean_val:.4e} ± {std_val:.4e} (n={len(values)})")
        else:
            print(f"  Avg {metric_name.replace('_', ' '):<10}: N/A (no data)")

    summary_csv_path = output_dir / f"validation_summary_fullmesh_{args.model_name}.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "count"])
        for metric_name, values in all_metrics.items():
            if values:
                writer.writerow([metric_name, avg_metrics_summary[f"{metric_name}_mean"], avg_metrics_summary[f"{metric_name}_std"], len(values)])
    print(f"Saved summary metrics to: {summary_csv_path}")

    if wandb_run:
        wandb_run.log(avg_metrics_summary)
        wandb_run.finish()

    print("\nFull mesh validation script finished.")

if __name__ == "__main__":
    main()
