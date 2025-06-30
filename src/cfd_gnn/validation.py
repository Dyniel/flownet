# -*- coding: utf-8 -*-
"""
validation.py
-------------
Functions for standalone model validation, including histogram-based JSD analysis
and inference with metric computation on specified validation datasets.
"""
import os
import glob
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import meshio
import wandb # For optional logging within these functions

from .data_utils import vtk_to_knn_graph, vtk_to_fullmesh_graph, sort_frames_by_number
from .metrics import compute_jsd_histograms, cosine_similarity_metric # TKE could also be added if needed
from .losses import calculate_divergence # For divergence metric
from .utils import write_vtk_with_fields, get_device

# --------------------------------------------------------------------- #
# Velocity Series Loading for Histogram Analysis
# --------------------------------------------------------------------- #

def load_velocity_series(
    vtk_files: list[Path],
    velocity_key: str = "U",
    interpolate_to_points: np.ndarray | None = None, # Target points [N_target, 3] for interpolation
    progress_interval: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads a time series of velocity fields from VTK files.
    Optionally interpolates velocities to a common reference point set.

    Args:
        vtk_files: Sorted list of Paths to VTK files.
        velocity_key: Key for the velocity data in point_data.
        interpolate_to_points: If provided, velocity fields from each VTK are interpolated
                               to these target points using nearest neighbor.
                               The first VTK's points are used if None and series varies.
        progress_interval: How often to print progress.

    Returns:
        A tuple (velocities_series, reference_points):
        - velocities_series: NumPy array of shape [num_time_steps, num_points, 3].
        - reference_points: NumPy array of shape [num_points, 3] (either interpolate_to_points
                            or points from the first frame if no interpolation target given).
    """
    if not vtk_files:
        raise ValueError("vtk_files list is empty.")

    num_time_steps = len(vtk_files)
    velocities_list = []

    first_mesh = meshio.read(str(vtk_files[0]))
    if velocity_key not in first_mesh.point_data:
        raise KeyError(f"Velocity key '{velocity_key}' not found in {vtk_files[0]}.")

    reference_points = interpolate_to_points if interpolate_to_points is not None else first_mesh.points.astype(np.float32)
    num_ref_points = reference_points.shape[0]

    print(f"Loading velocity series for key '{velocity_key}' from {num_time_steps} files. Target points: {num_ref_points}")

    for t_idx, vtk_path in enumerate(vtk_files):
        mesh = meshio.read(str(vtk_path))
        current_points = mesh.points.astype(np.float32)

        if velocity_key not in mesh.point_data:
            print(f"Warning: Velocity key '{velocity_key}' not found in {vtk_path}. Skipping frame.")
            # Add a placeholder or handle more robustly if strict continuity is needed.
            # For now, appending zeros if shapes match, else error.
            if len(velocities_list) > 0 and velocities_list[-1].shape[0] == num_ref_points :
                 velocities_list.append(np.zeros((num_ref_points, 3), dtype=np.float32))
            else: # Cannot determine shape for zeros, or first frame is missing key
                raise KeyError(f"Cannot proceed: Velocity key '{velocity_key}' missing in critical frame {vtk_path}")
            continue

        current_velocities = mesh.point_data[velocity_key].astype(np.float32)

        if interpolate_to_points is not None or (t_idx > 0 and not np.array_equal(current_points, reference_points)):
            # Interpolate current_velocities from current_points to reference_points
            if current_points.shape[0] == 0: # Handle empty mesh
                 print(f"Warning: Mesh {vtk_path} has 0 points. Appending zeros.")
                 interpolated_vel = np.zeros((num_ref_points, 3), dtype=np.float32)
            else:
                from sklearn.neighbors import NearestNeighbors # Lazy import
                nn = NearestNeighbors(n_neighbors=1).fit(current_points)
                _, nearest_indices = nn.kneighbors(reference_points)
                interpolated_vel = current_velocities[nearest_indices.squeeze()]
            velocities_list.append(interpolated_vel)
        else: # Points are consistent or it's the first frame without target interpolation points
            if current_velocities.shape[0] != num_ref_points:
                 print(f"Warning: Point count mismatch in {vtk_path} ({current_velocities.shape[0]}) vs reference ({num_ref_points}). Interpolating.")
                 if current_points.shape[0] == 0:
                     interpolated_vel = np.zeros((num_ref_points, 3), dtype=np.float32)
                 else:
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1).fit(current_points)
                    _, nearest_indices = nn.kneighbors(reference_points)
                    interpolated_vel = current_velocities[nearest_indices.squeeze()]
                 velocities_list.append(interpolated_vel)

            else:
                velocities_list.append(current_velocities)

        if (t_idx + 1) % progress_interval == 0 or (t_idx + 1) == num_time_steps:
            print(f"  ...loaded frame {t_idx + 1}/{num_time_steps}")

    return np.stack(velocities_list, axis=0), reference_points


# --------------------------------------------------------------------- #
# JSD Histogram Validation
# --------------------------------------------------------------------- #
@torch.no_grad()
def histogram_jsd_validation(
    model: torch.nn.Module | None, # Model can be None if predicted_data_root is provided
    cfg: dict, # Global config
    run_config: dict, # Specific config for this validation run (paths, model name)
    device: torch.device,
    wandb_run: wandb.sdk.wandb_run.Run | None = None # Optional W&B run object
):
    """
    Performs histogram-based JSD validation.
    Compares a reference dataset (e.g., original noisy data) against:
    1. Model predictions generated on-the-fly (if model provided and pred_data_root is None).
    2. Pre-computed predictions from predicted_data_root (if provided).

    Saves JSD heatmaps to VTK and logs metrics to W&B.

    Args:
        model: Trained GNN model (optional if predicted_data_root is used).
        cfg: Global configuration dictionary.
        run_config: Dictionary with run-specific settings:
            - "reference_data_root": Path to the reference dataset (e.g., CFD_Ubend_other_val_noisy).
            - "predicted_data_root": Path to pre-computed model predictions (optional). If None, model makes predictions.
            - "output_dir_jsd": Directory to save JSD VTK heatmaps.
            - "model_name_prefix": Prefix for W&B logging (e.g., "FlowNet_HistVal").
            - "graph_config_val": Graph construction parameters for on-the-fly prediction.
            - "graph_type_val": "knn" or "full_mesh" for on-the-fly prediction.
            - "velocity_key_ref": Velocity key in reference VTKs.
            - "velocity_key_pred": Velocity key in prediction VTKs.
        device: PyTorch device.
        wandb_run: Optional W&B run object for logging.
    """
    model_name_prefix = run_config.get("model_name_prefix", "Model")
    output_dir_jsd_base = Path(run_config["output_dir_jsd"])
    output_dir_jsd_base.mkdir(parents=True, exist_ok=True)

    reference_root = Path(run_config["reference_data_root"])
    predicted_root = Path(run_config["predicted_data_root"]) if run_config.get("predicted_data_root") else None

    if model is None and predicted_root is None:
        raise ValueError("Either a model or a predicted_data_root must be provided for JSD validation.")
    if model is not None:
        model.eval().to(device)

    # Iterate over cases found in the reference_data_root
    case_dirs_ref = sorted([d for d in reference_root.iterdir() if d.is_dir() and d.name.startswith("sUbend")])

    for case_dir_ref in case_dirs_ref:
        case_name = case_dir_ref.name
        print(f"\n--- JSD Validation for Case: {case_name} ({model_name_prefix}) ---")

        ref_cfd_dir = case_dir_ref / "CFD"
        if not ref_cfd_dir.is_dir():
            print(f"Reference CFD dir not found: {ref_cfd_dir}, skipping case.")
            continue

        ref_vtk_files = sort_frames_by_number(list(ref_cfd_dir.glob("*.vtk")))
        if not ref_vtk_files:
            print(f"No reference VTK files in {ref_cfd_dir}, skipping case.")
            continue

        # 1. Load reference velocity series
        # The points from the first reference frame will be our common grid for JSD.
        print(f"Loading reference velocity series from: {ref_cfd_dir}")
        ref_vel_series, ref_points_for_jsd = load_velocity_series(
            ref_vtk_files, velocity_key=run_config["velocity_key_ref"]
        )
        ref_mag_series = np.linalg.norm(ref_vel_series, axis=2) # [T, N]

        # 2. Obtain predicted velocity series
        pred_mag_series = None
        if predicted_root: # Use pre-computed predictions
            pred_cfd_dir = predicted_root / case_name / "CFD"
            if not pred_cfd_dir.is_dir():
                print(f"Predicted CFD dir not found: {pred_cfd_dir}, skipping case for JSD.")
                continue
            pred_vtk_files = sort_frames_by_number(list(pred_cfd_dir.glob("*.vtk")))
            if not pred_vtk_files or len(pred_vtk_files) != len(ref_vtk_files):
                print(f"Prediction VTK file mismatch in {pred_cfd_dir} (found {len(pred_vtk_files)}, ref {len(ref_vtk_files)}), skipping JSD.")
                continue

            print(f"Loading predicted velocity series from: {pred_cfd_dir}")
            # Interpolate predictions to the reference grid
            pred_vel_series, _ = load_velocity_series(
                pred_vtk_files,
                velocity_key=run_config["velocity_key_pred"],
                interpolate_to_points=ref_points_for_jsd
            )
            pred_mag_series = np.linalg.norm(pred_vel_series, axis=2)

        elif model: # Generate predictions on-the-fly
            print(f"Generating predictions on-the-fly for case {case_name} using {model_name_prefix}...")
            graph_cfg_val = run_config["graph_config_val"]
            graph_type_val = run_config["graph_type_val"]

            pred_vel_list_on_fly = []
            for vtk_path_ref in ref_vtk_files: # Predict for each reference frame
                if graph_type_val == "knn":
                    # Graph from ref data (can be noisy or clean, depends on what model expects)
                    graph_input = vtk_to_knn_graph(
                        vtk_path_ref, **graph_cfg_val,
                        use_noisy_data=graph_cfg_val.get("use_noisy_data", True), # Model usually trained on noisy
                        device=device
                    )
                elif graph_type_val == "full_mesh":
                    graph_input = vtk_to_fullmesh_graph(
                        vtk_path_ref, velocity_key=graph_cfg_val.get("velocity_key", "U"), # Key from input file
                        pressure_key=graph_cfg_val.get("pressure_key", "p"), device=device
                    )
                else: raise ValueError(f"Unknown graph type for on-the-fly JSD: {graph_type_val}")

                pred_vel_on_fly = model(graph_input).cpu().numpy() # [N_graph, 3]

                # Interpolate to ref_points_for_jsd if graph points differ
                current_graph_points = graph_input.pos.cpu().numpy()
                if not np.array_equal(current_graph_points, ref_points_for_jsd):
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1).fit(current_graph_points)
                    _, nearest_indices = nn.kneighbors(ref_points_for_jsd)
                    pred_vel_list_on_fly.append(pred_vel_on_fly[nearest_indices.squeeze()])
                else:
                    pred_vel_list_on_fly.append(pred_vel_on_fly)

            pred_vel_series_on_fly = np.stack(pred_vel_list_on_fly, axis=0)
            pred_mag_series = np.linalg.norm(pred_vel_series_on_fly, axis=2)

        if pred_mag_series is None:
            print(f"Could not obtain prediction series for {case_name}, skipping JSD computation.")
            continue

        # 3. Compute JSD
        print(f"Computing JSD for {case_name} (Ref shape: {ref_mag_series.shape}, Pred shape: {pred_mag_series.shape})...")
        if ref_mag_series.shape != pred_mag_series.shape:
            print(f"Shape mismatch for JSD: Ref {ref_mag_series.shape}, Pred {pred_mag_series.shape}. Skipping.")
            continue

        jsd_values, _, _ = compute_jsd_histograms(
            ref_mag_series, pred_mag_series, num_bins=cfg.get("nbins", cfg.get("BINS", 32))
        )

        # 4. Save JSD heatmap VTK
        output_dir_jsd_case = output_dir_jsd_base / model_name_prefix / case_name / "CFD"
        output_dir_jsd_case.mkdir(parents=True, exist_ok=True)
        jsd_vtk_path = output_dir_jsd_case / f"{case_name}_hist_jsd.vtk"

        write_vtk_with_fields(
            jsd_vtk_path,
            points=ref_points_for_jsd, # JSD is defined on these points
            point_data={"JSD": jsd_values.astype(np.float32)}
        )
        print(f"Saved JSD heatmap to: {jsd_vtk_path}")

        # 5. Log to W&B
        mean_jsd = float(np.mean(jsd_values))
        std_jsd = float(np.std(jsd_values))
        print(f"Case {case_name}: Mean JSD = {mean_jsd:.4e}, Std JSD = {std_jsd:.4e}")
        if wandb_run:
            wandb_run.log({
                f"{model_name_prefix}/{case_name}/JSD_mean": mean_jsd,
                f"{model_name_prefix}/{case_name}/JSD_std": std_jsd,
                # f"{model_name_prefix}/{case_name}/JSD_histogram": wandb.Histogram(jsd_values) # Log dist of JSD values
            })
            # Log example JSD VTK as artifact (optional, can be many files)
            # art = wandb.Artifact(f"jsd_heatmap_{model_name_prefix}_{case_name}", type="jsd_visualization")
            # art.add_file(str(jsd_vtk_path))
            # wandb_run.log_artifact(art)
    print(f"--- JSD Validation for {model_name_prefix} Complete ---")

# More functions for other validation types can be added here.

if __name__ == '__main__':
    from .models import FlowNet # Example model
    from .utils import set_seed

    print("Testing validation.py...")
    set_seed(42)
    test_device = get_device("auto")

    # Setup dummy data for testing histogram_jsd_validation
    dummy_val_root = Path("outputs/dummy_val_data_hist")
    dummy_pred_root = Path("outputs/dummy_pred_data_hist") # For pre-computed predictions
    dummy_jsd_out = Path("outputs/dummy_jsd_output")

    if dummy_val_root.exists(): shutil.rmtree(dummy_val_root)
    if dummy_pred_root.exists(): shutil.rmtree(dummy_pred_root)
    if dummy_jsd_out.exists(): shutil.rmtree(dummy_jsd_out)

    case1_val_cfd = dummy_val_root / "sUbend_val01" / "CFD"
    case1_val_cfd.mkdir(parents=True, exist_ok=True)
    case1_pred_cfd = dummy_pred_root / "sUbend_val01" / "CFD" # For pre-computed
    case1_pred_cfd.mkdir(parents=True, exist_ok=True)

    points_np = np.random.rand(25, 3).astype(np.float64)
    for i in range(3): # 3 frames
        # Reference data (e.g., noisy real)
        vel_ref = np.random.rand(25, 3).astype(np.float32) * (i+1) # Varying velocities
        ref_mesh = meshio.Mesh(points_np, point_data={"U_noisy": vel_ref, "U": vel_ref*0.8, "p": np.random.rand(25).astype(np.float32)})
        meshio.write(str(case1_val_cfd / f"Frame_{i:02d}_data.vtk"), ref_mesh)

        # Predicted data (if using pre-computed path)
        vel_pred_file = np.random.rand(25, 3).astype(np.float32) * (i+0.5)
        pred_mesh_file = meshio.Mesh(points_np, point_data={"velocity": vel_pred_file}) # Key "velocity"
        meshio.write(str(case1_pred_cfd / f"Frame_{i:02d}_pred.vtk"), pred_mesh_file)


    global_config_test = {"nbins": 16, "BINS": 16} # Example global config

    # --- Test with pre-computed predictions ---
    run_cfg_precomp = {
        "reference_data_root": str(dummy_val_root),
        "predicted_data_root": str(dummy_pred_root), # Provide this
        "output_dir_jsd": str(dummy_jsd_out),
        "model_name_prefix": "TestModel_Precomp",
        "velocity_key_ref": "U_noisy", # Key in reference files
        "velocity_key_pred": "velocity" # Key in prediction files
    }
    print("\nTesting JSD validation with pre-computed predictions...")
    histogram_jsd_validation(
        model=None, cfg=global_config_test, run_config=run_cfg_precomp, device=test_device
    )
    assert (dummy_jsd_out / "TestModel_Precomp" / "sUbend_val01" / "CFD" / "sUbend_val01_hist_jsd.vtk").exists()
    print("JSD validation with pre-computed predictions passed.")

    # --- Test with on-the-fly predictions ---
    # Dummy model
    model_cfg_test = {"h_dim": 16, "layers": 1}
    dummy_model_hist = FlowNet(model_cfg_test).to(test_device)

    graph_config_val_knn = { # For KNN graph building during on-the-fly prediction
        "k": 4, "down_n": None,
        "velocity_key": "U_noisy", # Model expects input from this key in ref files
        "use_noisy_data": True # This is for vtk_to_knn_graph to pick "U_noisy"
    }
    run_cfg_onfly = {
        "reference_data_root": str(dummy_val_root), # Model will read these, predict, then compare to these
        # "predicted_data_root": None, # Do not provide this to trigger on-the-fly
        "output_dir_jsd": str(dummy_jsd_out),
        "model_name_prefix": "TestModel_OnFly",
        "graph_config_val": graph_config_val_knn,
        "graph_type_val": "knn",
        "velocity_key_ref": "U_noisy", # For loading the reference series for comparison
        # "velocity_key_pred" is not used as predictions are direct arrays
    }
    print("\nTesting JSD validation with on-the-fly predictions...")
    histogram_jsd_validation(
        model=dummy_model_hist, cfg=global_config_test, run_config=run_cfg_onfly, device=test_device
    )
    assert (dummy_jsd_out / "TestModel_OnFly" / "sUbend_val01" / "CFD" / "sUbend_val01_hist_jsd.vtk").exists()
    print("JSD validation with on-the-fly predictions passed.")

    # Cleanup
    if dummy_val_root.exists(): shutil.rmtree(dummy_val_root)
    if dummy_pred_root.exists(): shutil.rmtree(dummy_pred_root)
    if dummy_jsd_out.exists(): shutil.rmtree(dummy_jsd_out)
    print("\nDummy validation test files cleaned up.")
    print("validation.py tests complete.")
