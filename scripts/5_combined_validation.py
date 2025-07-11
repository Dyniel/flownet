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
from src.cfd_gnn.metrics import calculate_vorticity_magnitude, calculate_velocity_gradients # For NS quantities
from src.cfd_gnn.point_slice_analysis import get_points_data, get_slice_data
import torch.nn.functional as F # For direct MSE
import pandas as pd

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
    # This will store lists of metrics, one list per case, each list containing metrics per frame
    metrics_per_case_then_frame = {
        "mse": {}, "rmse_mag": {}, "mse_div": {}, "cosine_sim": {},
        "mse_x": {}, "mse_y": {}, "mse_z": {},
        "max_true_vel_mag": {}, "max_pred_vel_mag": {} # Added max velocity
    }
    # This will store flat lists of metrics from all frames across all cases for global aggregation/histograms
    all_frames_metrics_flat = {key: [] for key in metrics_per_case_then_frame}

    # Setup CSV for per-frame metrics
    frame_metrics_csv_path = output_pred_dir.parent / f"frame_metrics_{args.model_name}_{args.graph_type}.csv"
    # Ensure output_pred_dir.parent exists, it's main_output_dir from main()
    # output_pred_dir is main_output_dir / "model_predictions"

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
    # The old slice_analysis (bounding box percentage based)
    old_slice_analysis_enabled = cfg.get("slice_analysis", {}).get("enabled", False)
    if old_slice_analysis_enabled:
        slice_metrics_csv_path = output_pred_dir.parent / f"frame_slice_metrics_old_{args.model_name}_{args.graph_type}.csv"
        slice_csv_file = open(slice_metrics_csv_path, "w", newline="")
        slice_csv_writer = csv.writer(slice_csv_file)
        slice_csv_header = [
            "case_name", "frame_file", "slice_axis", "slice_idx_in_axis",
            "slice_position", "num_points_in_slice",
            "max_true_vel_mag", "avg_true_vel_mag", "max_pred_vel_mag", "avg_pred_vel_mag"
        ]
        slice_csv_writer.writerow(slice_csv_header)
        print(f"Logging detailed per-frame-slice metrics (old method) to: {slice_metrics_csv_path}")
        all_slice_metrics_aggregated = {}

    # New Probe Analysis (Points and Slices)
    probes_config = cfg.get("analysis_probes", {})
    points_probe_config = probes_config.get("points", {})
    slices_probe_config = probes_config.get("slices", {})

    all_point_probe_data = [] # List to store dicts from get_points_data
    all_slice_probe_data = [] # List to store dicts from get_slice_data

    if points_probe_config.get("enabled", False) and not points_probe_config.get("coordinates"):
        print("Warning: Point probe analysis is enabled but no coordinates are defined. Disabling.")
        points_probe_config["enabled"] = False

    if slices_probe_config.get("enabled", False) and not slices_probe_config.get("definitions"):
        print("Warning: Slice probe analysis is enabled but no slice definitions are provided. Disabling.")
        slices_probe_config["enabled"] = False

    if points_probe_config.get("enabled", False):
        print(f"Point probe analysis enabled for {len(points_probe_config.get('coordinates',[]))} points.")
    if slices_probe_config.get("enabled", False):
        print(f"Slice probe analysis enabled for {len(slices_probe_config.get('definitions',[]))} slices.")

    print(f"Logging detailed per-frame metrics to: {frame_metrics_csv_path}")


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
        # Initialize lists for this case's frame metrics
        for metric_key in metrics_per_case_then_frame:
            metrics_per_case_then_frame[metric_key][case_name] = []

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

                # Write to detailed CSV
                detailed_csv_writer.writerow([
                    case_name, vtk_path.name, mse, rmse_mag, mse_div, cos_sim,
                    mse_x, mse_y, mse_z, max_true_vel_mag, max_pred_vel_mag
                ])

                # Calculate and log slice metrics if enabled (OLD METHOD)
                if old_slice_analysis_enabled and slice_csv_writer:
                    from src.cfd_gnn.metrics import calculate_slice_metrics_for_frame
                    slice_metrics_current_frame = calculate_slice_metrics_for_frame(
                        points_np=graph.pos.cpu().numpy(),
                        true_velocity_np=true_vel.cpu().numpy(),
                        pred_velocity_np=predicted_vel.cpu().numpy(),
                        slice_config=cfg.get("slice_analysis")
                    )
                    # ... (rest of the old slice analysis logging code remains the same) ...
                    for key, value in slice_metrics_current_frame.items():
                        parts = key.split('_')
                        if len(parts) >= 5 and parts[0] == "slice":
                            s_axis = parts[1]; s_idx = parts[2]; s_pos_str = parts[3].replace("pos","")
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

                # NEW PROBE ANALYSIS (Points and Slices)
                current_time = graph.time.item() if hasattr(graph, 'time') else np.nan
                mesh_points_np = graph.pos.cpu().numpy()
                true_vel_np = true_vel.cpu().numpy()
                pred_vel_np = predicted_vel.cpu().numpy()

                if points_probe_config.get("enabled", False):
                    target_coords = points_probe_config.get("coordinates", [])
                    true_points_data = get_points_data(mesh_points_np, target_coords, true_vel_np, current_time, "true_velocity")
                    pred_points_data = get_points_data(mesh_points_np, target_coords, pred_vel_np, current_time, "pred_velocity")

                    for i in range(len(true_points_data)):
                        # Merge true and pred data for the same target point
                        # Basic data from get_points_data: time, target_point_coords, actual_point_coords, distance_to_actual, true_velocity
                        merged_point_data = {
                            "case_name": case_name, "frame_file": vtk_path.name, **true_points_data[i]
                        }
                        merged_point_data["pred_velocity"] = pred_points_data[i].get("pred_velocity")

                        # Calculate Vorticity and Gradients for this one point
                        # Need to use the actual mesh point and its velocity for these calculations
                        point_coord_for_deriv = np.array([true_points_data[i]["actual_point_coords"]]) # Needs to be [1,3]
                        true_vel_at_point = np.array([true_points_data[i]["true_velocity"]]) # Needs to be [1,3]
                        pred_vel_at_point = np.array([pred_points_data[i]["pred_velocity"]]) if pred_points_data[i].get("pred_velocity") else np.array([[np.nan,np.nan,np.nan]])

                        merged_point_data["true_vort_mag"] = calculate_vorticity_magnitude(point_coord_for_deriv, true_vel_at_point)[0] if true_vel_at_point.ndim == 2 else np.nan
                        merged_point_data["pred_vort_mag"] = calculate_vorticity_magnitude(point_coord_for_deriv, pred_vel_at_point)[0] if pred_vel_at_point.ndim == 2 and not np.isnan(pred_vel_at_point).any() else np.nan

                        true_grads = calculate_velocity_gradients(point_coord_for_deriv, true_vel_at_point) # Shape [1,9] or None
                        pred_grads = calculate_velocity_gradients(point_coord_for_deriv, pred_vel_at_point) if pred_vel_at_point.ndim == 2 and not np.isnan(pred_vel_at_point).any() else None

                        grad_names = ['dudx', 'dudy', 'dudz', 'dvdx', 'dvdy', 'dvdz', 'dwdx', 'dwdy', 'dwdz']
                        for g_idx, g_name in enumerate(grad_names):
                            merged_point_data[f"true_grad_{g_name}"] = true_grads[0, g_idx] if true_grads is not None else np.nan
                            merged_point_data[f"pred_grad_{g_name}"] = pred_grads[0, g_idx] if pred_grads is not None else np.nan

                        # Calculate divergence: du/dx + dv/dy + dw/dz
                        merged_point_data["true_divergence"] = (true_grads[0,0] + true_grads[0,4] + true_grads[0,8]) if true_grads is not None else np.nan
                        merged_point_data["pred_divergence"] = (pred_grads[0,0] + pred_grads[0,4] + pred_grads[0,8]) if pred_grads is not None else np.nan

                        all_point_probe_data.append(merged_point_data)

                if slices_probe_config.get("enabled", False):
                    axis_map = {"X": 0, "Y": 1, "Z": 2}
                    for slice_def in slices_probe_config.get("definitions", []):
                        axis_name = slice_def.get("axis", "X").upper()
                        slice_axis_idx = axis_map.get(axis_name)
                        if slice_axis_idx is None:
                            print(f"Warning: Invalid slice axis '{axis_name}' in probe config. Skipping.")
                            continue

                        slice_pos = slice_def.get("position", 0.0)
                        slice_thick = slice_def.get("thickness", 0.01)

                        true_slice_info = get_slice_data(mesh_points_np, true_vel_np, current_time, slice_axis_idx, slice_pos, slice_thick)
                        pred_slice_info = get_slice_data(mesh_points_np, pred_vel_np, current_time, slice_axis_idx, slice_pos, slice_thick)

                        slice_data_row = {
                            "case_name": case_name, "frame_file": vtk_path.name, "time": current_time,
                            "slice_axis": axis_name, "slice_position": slice_pos, "slice_thickness": slice_thick,
                            "num_points_true": true_slice_info["num_points_in_slice"],
                            "avg_true_vel_mag": true_slice_info["avg_velocity_magnitude"],
                            "num_points_pred": pred_slice_info["num_points_in_slice"],
                            "avg_pred_vel_mag": pred_slice_info["avg_velocity_magnitude"]
                        }

                        # Vorticity and Gradients/Divergence for slice (average of point values in slice)
                        if true_slice_info["num_points_in_slice"] > 0:
                            true_vort_mag_slice = calculate_vorticity_magnitude(true_slice_info["points_in_slice"], true_slice_info["velocities_in_slice"])
                            slice_data_row["avg_true_vort_mag_slice"] = np.mean(true_vort_mag_slice) if true_vort_mag_slice is not None else np.nan

                            true_grads_slice = calculate_velocity_gradients(true_slice_info["points_in_slice"], true_slice_info["velocities_in_slice"])
                            if true_grads_slice is not None:
                                true_divergence_slice = true_grads_slice[:,0] + true_grads_slice[:,4] + true_grads_slice[:,8]
                                slice_data_row["avg_true_divergence_slice"] = np.mean(true_divergence_slice)
                                # Could average individual gradient components if needed: np.mean(true_grads_slice[:,i])
                            else:
                                slice_data_row["avg_true_divergence_slice"] = np.nan
                        else:
                            slice_data_row["avg_true_vort_mag_slice"] = np.nan
                            slice_data_row["avg_true_divergence_slice"] = np.nan

                        if pred_slice_info["num_points_in_slice"] > 0 and not np.isnan(pred_slice_info["velocities_in_slice"]).any():
                            pred_vort_mag_slice = calculate_vorticity_magnitude(pred_slice_info["points_in_slice"], pred_slice_info["velocities_in_slice"])
                            slice_data_row["avg_pred_vort_mag_slice"] = np.mean(pred_vort_mag_slice) if pred_vort_mag_slice is not None else np.nan

                            pred_grads_slice = calculate_velocity_gradients(pred_slice_info["points_in_slice"], pred_slice_info["velocities_in_slice"])
                            if pred_grads_slice is not None:
                                pred_divergence_slice = pred_grads_slice[:,0] + pred_grads_slice[:,4] + pred_grads_slice[:,8]
                                slice_data_row["avg_pred_divergence_slice"] = np.mean(pred_divergence_slice)
                            else:
                                slice_data_row["avg_pred_divergence_slice"] = np.nan
                        else:
                            slice_data_row["avg_pred_vort_mag_slice"] = np.nan
                            slice_data_row["avg_pred_divergence_slice"] = np.nan

                        all_slice_probe_data.append(slice_data_row)

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

    # After processing all frames for all cases, save collected probe data
    if points_probe_config.get("enabled", False) and all_point_probe_data:
        df_points = pd.DataFrame(all_point_probe_data)
        points_csv_path = output_pred_dir.parent / f"probed_points_data_{args.model_name}_{args.graph_type}.csv"
        df_points.to_csv(points_csv_path, index=False)
        print(f"Saved probed points data to: {points_csv_path}")
        if wandb_run:
            art_points_probe = wandb.Artifact(f"probed_points_data_{args.model_name}_{args.graph_type}", type="probed_points_timeseries")
            art_points_probe.add_file(str(points_csv_path))
            wandb_run.log_artifact(art_points_probe)

    if slices_probe_config.get("enabled", False) and all_slice_probe_data:
        df_slices = pd.DataFrame(all_slice_probe_data)
        slices_csv_path = output_pred_dir.parent / f"probed_slices_data_{args.model_name}_{args.graph_type}.csv"
        df_slices.to_csv(slices_csv_path, index=False)
        print(f"Saved probed slices data to: {slices_csv_path}")
        if wandb_run:
            art_slices_probe = wandb.Artifact(f"probed_slices_data_{args.model_name}_{args.graph_type}", type="probed_slices_timeseries")
            art_slices_probe.add_file(str(slices_csv_path))
            wandb_run.log_artifact(art_slices_probe)


    if old_slice_analysis_enabled and slice_csv_writer: # Close slice metrics CSV if it was opened
        slice_csv_file.close()

    detailed_csv_file.close() # Close the detailed CSV

    # --- Aggregate and Log Metrics ---
    # 1. Overall metrics (mean and std over all frames from all cases)
    summary_overall_metrics = {}
    print(f"\n  Overall Summary Metrics ({args.model_name}, Graph: {args.graph_type}, All Cases & Frames):")
    for metric_name, flat_values_list in all_frames_metrics_flat.items():
        if flat_values_list:
            mean_val = float(np.mean(flat_values_list))
            std_val = float(np.std(flat_values_list))
            summary_overall_metrics[f"Overall/{args.model_name}/{metric_name}_mean"] = mean_val
            summary_overall_metrics[f"Overall/{args.model_name}/{metric_name}_std"] = std_val
            print(f"    Avg {metric_name}: {mean_val:.4e} ± {std_val:.4e} (n={len(flat_values_list)})")
            if wandb_run: # Log histogram of all frame values for this metric
                 wandb_run.log({f"Histograms_All_Frames/{args.model_name}/{metric_name}": wandb.Histogram(np.array(flat_values_list))})

    # 2. Per-case summary metrics (mean and std over frames within each case)
    summary_per_case_metrics = {}
    print(f"\n  Per-Case Summary Metrics ({args.model_name}, Graph: {args.graph_type}):")
    for case_name_processed in case_names_to_process:
        print(f"    Case: {case_name_processed}")
        for metric_name, case_metric_data in metrics_per_case_then_frame.items(): # This loops through mse, rmse_mag etc.
            values_for_case = case_metric_data.get(case_name_processed, [])
            if values_for_case:
                mean_val = float(np.mean(values_for_case))
                std_val = float(np.std(values_for_case))
                # Log to a dictionary for W&B, namespaced by case
                summary_per_case_metrics[f"PerCase/{args.model_name}/{case_name_processed}/{metric_name}_mean"] = mean_val
                summary_per_case_metrics[f"PerCase/{args.model_name}/{case_name_processed}/{metric_name}_std"] = std_val
                print(f"      Avg {metric_name}: {mean_val:.4e} ± {std_val:.4e} (n={len(values_for_case)})")

    # 3. Aggregated Slice Metrics for W&B (OLD METHOD)
    summary_slice_metrics_wandb = {}
    if old_slice_analysis_enabled and all_slice_metrics_aggregated:
        print(f"\n  Aggregated Slice Metrics (Old Method) ({args.model_name}, Graph: {args.graph_type}):")
        for agg_key, values_list in all_slice_metrics_aggregated.items():
            if values_list:
                mean_val = float(np.mean(values_list))
                std_val = float(np.std(values_list))
                summary_slice_metrics_wandb[f"SliceMetricsOld/{args.model_name}/{agg_key}_mean"] = mean_val
                summary_slice_metrics_wandb[f"SliceMetricsOld/{args.model_name}/{agg_key}_std"] = std_val
                print(f"    {agg_key}: Mean={mean_val:.4e}, Std={std_val:.4e} (n={len(values_list)})")

    # (No specific W&B summary logging for new probe data here, as it's saved as artifacts)

    if wandb_run:
        if summary_overall_metrics:
            wandb_run.log(summary_overall_metrics)
        if summary_per_case_metrics:
            wandb_run.log(summary_per_case_metrics)
        if summary_slice_metrics_wandb: # For old slice metrics
            wandb_run.log(summary_slice_metrics_wandb)

        if old_slice_analysis_enabled and slice_metrics_csv_path and slice_metrics_csv_path.exists():
            art_slice_metrics = wandb.Artifact(f"frame_slice_metrics_old_{args.model_name}_{args.graph_type}", type="per_frame_slice_metrics_old")
            art_slice_metrics.add_file(str(slice_metrics_csv_path))
            wandb_run.log_artifact(art_slice_metrics)

    return summary_overall_metrics # Return overall summary for the main CSV


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
