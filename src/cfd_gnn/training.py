# -*- coding: utf-8 -*-
"""
training.py
-----------
Functions for training and validating CFD GNN models during the training process.
Includes epoch-based training loops and validation on continuous (paired) data.
"""
import time
import numpy as np
import torch
from pathlib import Path
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader  # Use PyG DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F  # For direct use if needed, though losses module is preferred

from .losses import combined_loss, calculate_divergence  # Assuming losses.py is in the same package
from .data_utils import vtk_to_knn_graph  # Or a more general graph loader if needed for validation
from .utils import get_device, write_vtk_with_fields  # For device management and VTK writing
from .metrics import calculate_vorticity_magnitude  # For vorticity calculation
from pathlib import Path  # For path manipulation
import numpy as np  # For array operations if needed before PyVista
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import io
import wandb
from PIL import Image # Import PIL Image

# New helper function for plotting:
def _log_and_save_field_plots(
    points_np: np.ndarray,
    true_vel_tensor: torch.Tensor,
    pred_vel_tensor: torch.Tensor,
    error_mag_tensor: torch.Tensor,
    pred_div_tensor: torch.Tensor,
    # pred_vort_mag_np: np.ndarray | None, # Removed vorticity from plots for now
    path_t1: Path, # For naming output files
    epoch_num: int,
    current_sample_global_idx: int, # Index of the sample in the validation dataloader
    output_base_dir: str | Path | None,
    wandb_run: wandb.sdk.wandb_run.Run | None,
    model_name: str,
    # base_filename_stem: str, # path_t1.stem can be used directly
    simple_plot: bool = False # Flag for the fallback simpler plot
):
    """
    Generates, saves, and logs detailed field comparison plots for a validation sample.
    Plots include velocity magnitudes (true, pred, error), predicted divergence,
    and predicted vorticity magnitude.
    """
    if not (wandb_run or output_base_dir): # No place to log or save
        return
    if points_np is None or points_np.shape[0] < 3: # Need at least 3 points for triangulation
        print(f"Warning: Not enough points ({points_np.shape[0] if points_np is not None else 'None'}) for sample {current_sample_global_idx} from {path_t1.name} to generate field plot.")
        return

    # Convert tensors to numpy for plotting
    true_vel_mag_np = true_vel_tensor.norm(dim=1).cpu().numpy()
    pred_vel_mag_np = pred_vel_tensor.norm(dim=1).cpu().numpy()
    error_mag_np = error_mag_tensor.cpu().numpy()
    pred_div_np = pred_div_tensor.cpu().numpy() if pred_div_tensor is not None else None # Handle optional div

    # Determine slice for 2D plotting (e.g., points near z=mean(z) or just XY if 2D)
    slice_points_2d = None
    fields_to_plot_on_slice = {}

    is_3d_data = points_np.shape[1] == 3

    if is_3d_data:
        mean_z = np.mean(points_np[:, 2])
        z_extent = np.max(points_np[:, 2]) - np.min(points_np[:, 2])
        thickness = 0.05 * z_extent if z_extent > 1e-6 else 0.01
        slice_indices = np.where(np.abs(points_np[:, 2] - mean_z) < thickness / 2.0)[0]

        if len(slice_indices) < 3:
            slice_points_2d = points_np[:, :2]
            slice_indices = np.arange(points_np.shape[0])
        else:
            slice_points_2d = points_np[slice_indices, :2]
    else: # Data is already 2D
        slice_points_2d = points_np
        slice_indices = np.arange(points_np.shape[0])

    if len(slice_indices) < 3:
        print(f"Warning: Not enough points in final slice ({len(slice_indices)} points) for sample {current_sample_global_idx} from {path_t1.name}. Skipping plot generation.")
        return

    # Populate fields for plotting
    fields_to_plot_on_slice["True Vel Mag"] = true_vel_mag_np[slice_indices]
    fields_to_plot_on_slice["Pred Vel Mag"] = pred_vel_mag_np[slice_indices]
    fields_to_plot_on_slice["Error Mag"] = error_mag_np[slice_indices]

    if not simple_plot:
        if pred_div_np is not None:
            fields_to_plot_on_slice["Pred Divergence"] = pred_div_np[slice_indices]
        # Vorticity plotting removed
        # if pred_vort_mag_np is not None and is_3d_data:
        #      if pred_vort_mag_np.shape[0] == points_np.shape[0]:
        #         fields_to_plot_on_slice["Pred Vorticity Mag"] = pred_vort_mag_np[slice_indices]
        #      else:
        #         print(f"Warning: Vorticity array shape mismatch for sample {current_sample_global_idx} from {path_t1.name}. Skipping vorticity plot.")

    num_subplots = len(fields_to_plot_on_slice)
    if num_subplots == 0:
        return

    if simple_plot and "Error Mag" not in fields_to_plot_on_slice: # Ensure simple plot has at least error
        if "Pred Vel Mag" in fields_to_plot_on_slice : del fields_to_plot_on_slice["Pred Vel Mag"] # make space
        if "True Vel Mag" in fields_to_plot_on_slice : del fields_to_plot_on_slice["True Vel Mag"]
        num_subplots = len(fields_to_plot_on_slice)


    if num_subplots <= 3 :
        fig_rows, fig_cols = 1, num_subplots
        figsize = (6 * num_subplots, 5)
    else:
        fig_cols = 3
        fig_rows = (num_subplots + fig_cols -1) // fig_cols
        figsize = (18, 5 * fig_rows) if fig_rows > 0 else (18,5)


    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    plot_successful = False
    try:
        tri = Delaunay(slice_points_2d)

        for ax_idx, (title, data_field) in enumerate(fields_to_plot_on_slice.items()):
            ax = axes[ax_idx]
            cmap = "jet"
            if "Error" in title: cmap = "Reds"
            elif "Divergence" in title: cmap = "coolwarm"
            elif "Vorticity" in title: cmap = "viridis"

            contour = ax.tricontourf(slice_points_2d[:,0], slice_points_2d[:,1], tri.simplices, data_field, levels=14, cmap=cmap)
            fig.colorbar(contour, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.axis('equal')

        for ax_idx in range(num_subplots, len(axes)):
            axes[ax_idx].set_visible(False)

        plot_title_prefix = f"Epoch {epoch_num}" if epoch_num >=0 else "FinalVal"
        fig.suptitle(f"{model_name} - {plot_title_prefix} - Sample {current_sample_global_idx} ({path_t1.stem}) - {'Z-slice' if is_3d_data else '2D'} Fields", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plot_successful = True

    except Exception as e_plot:
        print(f"Warning: Failed to generate tricontourf plot for sample {current_sample_global_idx} from {path_t1.name}: {e_plot}")
        if fig is not None: plt.close(fig)
        return

    if plot_successful:
        if output_base_dir:
            try:
                case_name = path_t1.parent.parent.name
                epoch_folder_name = f"epoch_{epoch_num}" if epoch_num >= 0 else "final_validation_plots"

                plot_output_dir = Path(output_base_dir) / "validation_plots" / model_name / epoch_folder_name / case_name
                plot_output_dir.mkdir(parents=True, exist_ok=True)

                base_filename = path_t1.stem
                plot_suffix = "_simple_comparison.png" if simple_plot else "_detailed_fields_comparison.png"
                plot_file_path = plot_output_dir / f"{base_filename}_sample{current_sample_global_idx}{plot_suffix}"

                plt.savefig(plot_file_path)
                # print(f"  Saved field plot to {plot_file_path}") # Reduced verbosity
            except Exception as e_save:
                print(f"Warning: Could not save local field plot for sample {current_sample_global_idx} from {path_t1.name}: {e_save}")

        if wandb_run:
            try:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                log_key_suffix = "_simple" if simple_plot else "_detailed"
                # Key no longer contains epoch number for W&B media panel slider
                image_log_key = f"{model_name}/Validation_Fields_Sample{current_sample_global_idx}{log_key_suffix}"
                pil_image = Image.open(buf)
                # Log with step=epoch_num and commit=False, so it's part of the main epoch log
                wandb_run.log({image_log_key: wandb.Image(pil_image)}, step=epoch_num, commit=False)
                buf.close()
                pil_image.close()

            except Exception as e_wandb_log:
                print(f"Warning: Could not log W&B field image for sample {current_sample_global_idx} from {path_t1.name} (epoch {epoch_num}): {e_wandb_log}")

        plt.close(fig)


def train_single_epoch(
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        loss_weights: dict,  # e.g., {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
        histogram_bins: int,
        device: torch.device,
        clip_grad_norm_value: float | None = 1.0,
        regularization_type: str = "None", # "L1", "L2", or "None"
        regularization_lambda: float = 0.0 # Strength of regularization
) -> dict:
    """
    Trains the model for a single epoch.

    Args:
        model: The GNN model to train.
        train_loader: DataLoader for the training dataset (yielding pairs of Data objects).
        optimizer: The PyTorch optimizer.
        loss_weights: Dictionary of weights for different loss components.
        histogram_bins: Number of bins for the histogram loss.
        device: PyTorch device to run training on.
        clip_grad_norm_value: Max norm for gradient clipping. If None, no clipping.

    Returns:
        A dictionary containing aggregated training metrics for the epoch (e.g., mean losses).
    """
    model.train()
    epoch_aggregated_losses = {
        "total": 0.0,
        "supervised": 0.0,
        "divergence": 0.0,
        "histogram": 0.0,
        "regularization": 0.0 # Added for L1/L2 regularization loss
    }
    num_batches = 0

    for graph_t0, graph_t1 in train_loader:
        # Move data to the target device within the training loop
        graph_t0 = graph_t0.to(device)
        graph_t1 = graph_t1.to(device)

        optimizer.zero_grad()

        # Forward pass: model predicts velocity at t1 based on graph_t0
        # The model's task is to predict graph_t1.x (true_vel_t1) using graph_t0 as input.
        predicted_vel_t1 = model(graph_t0)
        true_vel_t1 = graph_t1.x

        # graph_data for loss calculation should be graph_t0, as divergence is on predicted field
        # relative to the input graph structure.
        total_loss, individual_losses = combined_loss(
            predicted_velocity=predicted_vel_t1,
            true_velocity=true_vel_t1,
            graph_data=graph_t0,  # Divergence and histogram loss use the input graph's structure
            loss_weights=loss_weights,
            histogram_bins=histogram_bins
        )

        # Add regularization loss if applicable
        reg_loss = torch.tensor(0.0, device=device)
        if regularization_type != "None" and regularization_lambda > 0:
            if regularization_type == "L1":
                for param in model.parameters():
                    reg_loss += torch.sum(torch.abs(param))
            elif regularization_type == "L2":
                for param in model.parameters():
                    reg_loss += torch.sum(param.pow(2))
            else:
                raise ValueError(f"Unknown regularization_type: {regularization_type}. Supported types are 'L1', 'L2', 'None'.")

            total_loss += regularization_lambda * reg_loss
            epoch_aggregated_losses["regularization"] += (regularization_lambda * reg_loss).item()


        total_loss.backward()

        if clip_grad_norm_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)

        optimizer.step()

        epoch_aggregated_losses["total"] += total_loss.item()
        epoch_aggregated_losses["supervised"] += individual_losses["supervised"].item()
        epoch_aggregated_losses["divergence"] += individual_losses["divergence"].item()
        epoch_aggregated_losses["histogram"] += individual_losses["histogram"].item()

        # DEBUG: Log divergence and prediction stats for the first few batches
        if num_batches < 2: # Log for first 2 batches of an epoch
            if "divergence_values_pred_for_debug" in individual_losses:
                div_pred_train = individual_losses["divergence_values_pred_for_debug"]
                div_pred_stats_train = {
                    "min": div_pred_train.min().item(), "max": div_pred_train.max().item(),
                    "mean": div_pred_train.mean().item(), "std": div_pred_train.std().item(),
                    "abs_mean": div_pred_train.abs().mean().item()
                }
                print(f"DEBUG: Train batch {num_batches}, div_pred stats: {div_pred_stats_train}")
                print(f"DEBUG: Train batch {num_batches}, individual_losses[\"divergence\"]: {individual_losses['divergence'].item():.12e}") # High precision print

            pred_vel_stats_train = {
                "min": predicted_vel_t1.min().item(), "max": predicted_vel_t1.max().item(),
                "mean": predicted_vel_t1.mean().item(), "std": predicted_vel_t1.std().item(),
                "abs_mean": predicted_vel_t1.abs().mean().item()
            }
            print(f"DEBUG: Train batch {num_batches}, pred_vel_t1 stats: {pred_vel_stats_train}")

        num_batches += 1

    if num_batches > 0:
        for key in epoch_aggregated_losses:
            epoch_aggregated_losses[key] /= num_batches

    return epoch_aggregated_losses


@torch.no_grad()
def validate_on_pairs(
        model: torch.nn.Module,
        val_frame_pairs: list[tuple[Path, Path]],  # List of (path_t0, path_t1)
        # graph_config: dict,  # For vtk_to_graph (k_neighbors, downsample_n, keys) # Replaced by global_cfg access
        # For probe points, we need the global config, not just graph_config
        global_cfg: dict, # Main config dictionary
        use_noisy_data_for_val: bool,  # Whether val data itself is noisy
        device: torch.device,
        # graph_type: str = "knn",  # "knn" or "full_mesh" for graph construction # Get from global_cfg
        epoch_num: int = -1,  # For naming output files, -1 for non-epoch specific validation
        output_base_dir: str | Path | None = None,  # Base path for saving field VTKs
        save_fields_vtk: bool = False,  # Flag to control saving of VTK files
        wandb_run: wandb.sdk.wandb_run.Run | None = None,  # For logging images
        log_field_image_sample_idx: int = 0,  # Index of the sample in val_frame_pairs to log as an image
        model_name: str = "Model"  # For naming W&B logs
) -> tuple[dict, list]: # Will now also return probe_data_collected
    """
    Validates the model on a set of paired frames from VTK files.
    Computes MSE, RMSE of velocity magnitude, and MSE of divergence.
    Optionally saves VTK files with true, predicted, and error fields.
    Optionally logs velocity at probe points.

    Args:
        model: The trained GNN model.
        val_frame_pairs: List of (path_t0, path_t1) tuples for validation.
        global_cfg: The global configuration dictionary.
        use_noisy_data_for_val: If True, loads noisy versions of validation data.
        device: PyTorch device.
        epoch_num: Current epoch number, used for naming output directories for VTK fields and probe CSV.
        output_base_dir: Base directory for saving run outputs.
        save_fields_vtk: If True, saves the VTK files with detailed fields.
        wandb_run: Optional W&B run object.
        log_field_image_sample_idx: Index of sample for detailed field plot.
        model_name: Name of the model for logging.

    Returns:
        A tuple: (dictionary_with_mean_validation_metrics, list_of_probe_data_rows)
    """
    model.eval()
    metrics_list = { # Standard metrics
        "mse": [], "rmse_mag": [], "mse_div": [],
        "mse_x": [], "mse_y": [], "mse_z": [],
        # "mse_vorticity_mag": [], # Removed
        "cosine_sim": [],
        "max_true_vel_mag": [],
        "max_pred_vel_mag": []
    }
    # --- Probe Data Initialization ---
    probe_data_collected = [] # List to store dicts for CSV/Pandas for the detailed CSV file
    # Data for W&B Table: list of lists/tuples: [case, probe_id, target_x, target_y, target_z, error_mag]
    wandb_table_probe_errors_data = []
    case_probe_definitions = {} # Stores {case_name: [(target_coord_str, node_idx, target_coord_xyz), ...]}

    from .data_utils import vtk_to_knn_graph, vtk_to_fullmesh_graph # Local import
    from .metrics import cosine_similarity_metric # Import cosine similarity
    from sklearn.neighbors import NearestNeighbors # For probe point finding

    # Extract relevant configs from global_cfg
    # Use validation_during_training specific graph_config if available, else main graph_config
    # This was simplified from the original as global_cfg is now passed directly.
    val_train_cfg = global_cfg.get("validation_during_training", {})
    graph_config = val_train_cfg.get("val_graph_config", global_cfg.get("graph_config", {}))
    # Ensure essential keys are present, falling back to top-level cfg or defaults
    graph_config.update({
        "k": graph_config.get("k", global_cfg.get("graph_config", {}).get("k", 12)), # Default k if not found anywhere
        "down_n": graph_config.get("down_n", global_cfg.get("graph_config", {}).get("down_n")),
        "velocity_key": graph_config.get("velocity_key", global_cfg.get("velocity_key", "U")),
        "noisy_velocity_key_suffix": graph_config.get("noisy_velocity_key_suffix", global_cfg.get("noisy_velocity_key_suffix", "_noisy")),
        "pressure_key": graph_config.get("pressure_key", global_cfg.get("pressure_key", "p"))
    })
    graph_type = val_train_cfg.get("val_graph_type", global_cfg.get("default_graph_type", "knn"))

    probes_cfg = global_cfg.get("analysis_probes", {}).get("points", {})
    probes_enabled = probes_cfg.get("enabled", False)
    # Default to [1,1,1] (center point) if num_probes_per_axis not specified
    probes_num_per_axis = probes_cfg.get("num_probes_per_axis", [1,1,1])
    probes_output_field_name = probes_cfg.get("output_field_name", "velocity_at_probe")


    for i, (path_t0, path_t1) in enumerate(val_frame_pairs):
        current_case_name = path_t0.parent.parent.name # e.g. sUbend_011

        # --- Setup Probes for the current case if not already done ---
        if probes_enabled and current_case_name not in case_probe_definitions:
            print(f"DEBUG: Setting up probes for case: {current_case_name}")
            # To get geometry for probe placement, load the first frame (path_t0) of this case
            # This requires constructing a temporary graph object for its 'pos' attribute.
            temp_graph_args_for_pos = {
                "k_neighbors": graph_config["k"],
                "downsample_n": graph_config.get("down_n"),
                "velocity_key": graph_config.get("velocity_key"), # Use the determined velocity key
                "noisy_velocity_key_suffix": graph_config.get("noisy_velocity_key_suffix"),
                "use_noisy_data": use_noisy_data_for_val # Use validation noise setting
            }
            if graph_type == "knn":
                 first_frame_graph_cpu = vtk_to_knn_graph(path_t0, **temp_graph_args_for_pos)
            else: # full_mesh
                 first_frame_graph_cpu = vtk_to_fullmesh_graph(
                     path_t0,
                     velocity_key=graph_config.get("velocity_key"),
                     pressure_key=graph_config.get("pressure_key")
                 )

            case_points_np = first_frame_graph_cpu.pos.cpu().numpy()
            if case_points_np.shape[0] == 0:
                print(f"Warning: Case {current_case_name} first frame has no points. Skipping probe setup for this case.")
                case_probe_definitions[current_case_name] = [] # Mark as processed, no probes
            else:
                min_coords = case_points_np.min(axis=0)
                max_coords = case_points_np.max(axis=0)
                bbox_extents = max_coords - min_coords

                current_case_probes_list = []
                nx, ny, nz = probes_num_per_axis[0], probes_num_per_axis[1], probes_num_per_axis[2]

                for ix_prog in range(nx): # Use different loop var name
                    # Position is at (index + 1) / (count + 1) fraction of extent
                    px = min_coords[0] + (bbox_extents[0] * (ix_prog + 1) / (nx + 1)) if nx > 0 and bbox_extents[0] > 1e-6 else min_coords[0] + 0.5 * bbox_extents[0]
                    for iy_prog in range(ny):
                        py = min_coords[1] + (bbox_extents[1] * (iy_prog + 1) / (ny + 1)) if ny > 0 and bbox_extents[1] > 1e-6 else min_coords[1] + 0.5 * bbox_extents[1]
                        for iz_prog in range(nz):
                            pz = min_coords[2] + (bbox_extents[2] * (iz_prog + 1) / (nz + 1)) if nz > 0 and bbox_extents[2] > 1e-6 else min_coords[2] + 0.5 * bbox_extents[2]

                            target_coord = np.array([px, py, pz])
                            nn_probes = NearestNeighbors(n_neighbors=1).fit(case_points_np) # Renamed nn
                            _, nearest_node_idx_arr = nn_probes.kneighbors(target_coord.reshape(1, -1)) # Renamed
                            node_idx = int(nearest_node_idx_arr.squeeze())

                            target_coord_str = f"P_X{px:.2f}_Y{py:.2f}_Z{pz:.2f}"
                            current_case_probes_list.append((target_coord_str, node_idx, target_coord))
                case_probe_definitions[current_case_name] = current_case_probes_list
                print(f"DEBUG: Defined {len(current_case_probes_list)} probes for case {current_case_name}.")
            del first_frame_graph_cpu # Free memory


        # Graph construction (actual graphs for model input)
        if graph_type == "knn":
            knn_args_val = {
                "k_neighbors": graph_config["k"],
                "downsample_n": graph_config.get("down_n"),
                "velocity_key": graph_config.get("velocity_key"),
                "noisy_velocity_key_suffix": graph_config.get("noisy_velocity_key_suffix"),
            }
            graph_t0_cpu = vtk_to_knn_graph(path_t0, **knn_args_val, use_noisy_data=use_noisy_data_for_val)
            graph_t1_cpu = vtk_to_knn_graph(path_t1, **knn_args_val, use_noisy_data=use_noisy_data_for_val) # Target uses same noise setting
        elif graph_type == "full_mesh":
            graph_t0_cpu = vtk_to_fullmesh_graph(path_t0, velocity_key=graph_config.get("velocity_key"), pressure_key=graph_config.get("pressure_key"))
            graph_t1_cpu = vtk_to_fullmesh_graph(path_t1, velocity_key=graph_config.get("velocity_key"), pressure_key=graph_config.get("pressure_key"))
        else:
            raise ValueError(f"Unsupported graph_type for validation: {graph_type}")

        # Move graphs to device before model inference
        graph_t0 = graph_t0_cpu.to(device)
        graph_t1 = graph_t1_cpu.to(device)

        predicted_vel_t1 = model(graph_t0)
        true_vel_t1 = graph_t1.x

        # --- Standard Metrics Calculation ---
        mse = F.mse_loss(predicted_vel_t1, true_vel_t1).item()
        metrics_list["mse"].append(mse)
        for component_idx, component_label in enumerate(['x', 'y', 'z']):
            if predicted_vel_t1.shape[1] > component_idx: # Ensure component exists (e.g. for 2D data)
                mse_comp = F.mse_loss(predicted_vel_t1[:, component_idx], true_vel_t1[:, component_idx]).item()
                metrics_list[f"mse_{component_label}"].append(mse_comp)
            else:
                metrics_list[f"mse_{component_label}"].append(np.nan)


        pred_mag = predicted_vel_t1.norm(dim=1)
        true_mag = true_vel_t1.norm(dim=1)
        rmse_mag = torch.sqrt(F.mse_loss(pred_mag, true_mag)).item()
        metrics_list["rmse_mag"].append(rmse_mag)

        div_pred_tensor = calculate_divergence(predicted_vel_t1, graph_t0) # Renamed to avoid conflict
        mse_div = (div_pred_tensor ** 2).mean().item()
        metrics_list["mse_div"].append(mse_div)

        cos_sim = cosine_similarity_metric(predicted_vel_t1.cpu().numpy(), true_vel_t1.cpu().numpy()) # Numpy for metric
        metrics_list["cosine_sim"].append(cos_sim)

        max_true_vel_mag = true_mag.max().item()
        max_pred_vel_mag = pred_mag.max().item()
        metrics_list["max_true_vel_mag"].append(max_true_vel_mag)
        metrics_list["max_pred_vel_mag"].append(max_pred_vel_mag)

        points_np_frame = graph_t1.pos.cpu().numpy() # Points for current frame

        # Vorticity calculation and metric removed
        # if points_np_frame.shape[1] == 3: # Only if 3D data
        #     true_vort_mag_np = calculate_vorticity_magnitude(points_np_frame, true_vel_t1.cpu().numpy())
        #     pred_vort_mag_np = calculate_vorticity_magnitude(points_np_frame, predicted_vel_t1.cpu().numpy())
        #     if true_vort_mag_np is not None and pred_vort_mag_np is not None and \
        #        true_vort_mag_np.shape == pred_vort_mag_np.shape:
        #         mse_vort_mag = np.mean((true_vort_mag_np - pred_vort_mag_np) ** 2)
        #         metrics_list["mse_vorticity_mag"].append(mse_vort_mag)
        #     else: metrics_list["mse_vorticity_mag"].append(np.nan)
        # else: metrics_list["mse_vorticity_mag"].append(np.nan)

        # --- Probe Data Extraction and Logging ---
        if probes_enabled and current_case_name in case_probe_definitions and case_probe_definitions[current_case_name]:
            frame_time_val = graph_t0.time.item() if hasattr(graph_t0, 'time') and graph_t0.time is not None else float(i)

            for probe_idx, (target_coord_str, node_idx, target_coord_actual_xyz) in enumerate(case_probe_definitions[current_case_name]):
                if 0 <= node_idx < true_vel_t1.shape[0]: # Check if node_idx is valid for current graph
                    true_probe_vel_vals = true_vel_t1[node_idx].cpu().numpy()
                    pred_probe_vel_vals = predicted_vel_t1[node_idx].cpu().numpy()

                    true_probe_mag_val = np.linalg.norm(true_probe_vel_vals)
                    pred_probe_mag_val = np.linalg.norm(pred_probe_vel_vals)
                    error_vec_probe = true_probe_vel_vals - pred_probe_vel_vals
                    error_probe_mag_val = np.linalg.norm(error_vec_probe)

                    # Data for W&B Table
                    wandb_table_probe_errors_data.append([
                        current_case_name,
                        target_coord_str,
                        target_coord_actual_xyz[0],
                        target_coord_actual_xyz[1],
                        target_coord_actual_xyz[2],
                        error_probe_mag_val
                    ])

                    # CSV Data Collection (detailed)
                    probe_data_collected.append({
                        "epoch": epoch_num, "case": current_case_name,
                        "frame_path_t0": str(path_t0), "frame_time_t0": frame_time_val,
                        "probe_id_str": target_coord_str,
                        "probe_target_x": target_coord_actual_xyz[0],
                        "probe_target_y": target_coord_actual_xyz[1],
                        "probe_target_z": target_coord_actual_xyz[2],
                        "node_idx": node_idx,
                        "true_vx": true_probe_vel_vals[0], "true_vy": true_probe_vel_vals[1], "true_vz": true_probe_vel_vals[2] if len(true_probe_vel_vals) == 3 else 0.0,
                        "pred_vx": pred_probe_vel_vals[0], "pred_vy": pred_probe_vel_vals[1], "pred_vz": pred_probe_vel_vals[2] if len(pred_probe_vel_vals) == 3 else 0.0,
                        "true_mag": true_probe_mag_val, "pred_mag": pred_probe_mag_val, "error_mag": error_probe_mag_val
                    })
        # --- End Probe Data ---


        # --- VTK Saving and Plotting (using points_np_frame) ---
        if save_fields_vtk and output_base_dir and points_np_frame is not None:
            try:
                error_mag_for_vtk = torch.norm(predicted_vel_t1 - true_vel_t1, dim=1)
                frame_name_stem = path_t1.stem
                # case_name is current_case_name
                epoch_folder_name = f"epoch_{epoch_num}" if epoch_num >= 0 else "final_validation"
                vtk_output_dir = Path(output_base_dir) / "validation_fields" / model_name / epoch_folder_name / current_case_name
                vtk_output_dir.mkdir(parents=True, exist_ok=True)
                vtk_file_path = vtk_output_dir / f"{frame_name_stem}_fields.vtk"

                true_vel_np = true_vel_t1.cpu().numpy()
                pred_vel_np = predicted_vel_t1.cpu().numpy()
                delta_v_vectors = true_vel_np - pred_vel_np

                point_data_for_vtk = {
                    "true_velocity": true_vel_np,
                    "predicted_velocity": pred_vel_np,
                    "delta_velocity_vector": delta_v_vectors, # Add delta_v vector field
                    "velocity_error_magnitude": error_mag_for_vtk.cpu().numpy()
                }
                # Vorticity fields removed from VTK output
                # if points_np_frame.shape[1] == 3: # Vorticity only for 3D
                #     # true_vort_mag_np and pred_vort_mag_np already computed if 3D
                #     if true_vort_mag_np is not None: point_data_for_vtk["true_vorticity_magnitude"] = true_vort_mag_np
                #     if pred_vort_mag_np is not None: point_data_for_vtk["predicted_vorticity_magnitude"] = pred_vort_mag_np


                write_vtk_with_fields(str(vtk_file_path), points_np_frame, point_data_for_vtk)
            except Exception as e_vtk:
                print(f"Warning: Could not save detailed VTK fields for {path_t1.name}: {e_vtk}")

        if i == log_field_image_sample_idx and points_np_frame is not None and points_np_frame.shape[0] > 0:
            error_mag_tensor_for_plot = torch.norm(predicted_vel_t1 - true_vel_t1, dim=1)
            # Vorticity data removed from this call
            _log_and_save_field_plots(
                points_np=points_np_frame, true_vel_tensor=true_vel_t1, pred_vel_tensor=predicted_vel_t1,
                error_mag_tensor=error_mag_tensor_for_plot, pred_div_tensor=div_pred_tensor,
                # pred_vort_mag_np=pred_vort_mag_np if points_np_frame.shape[1] == 3 else None, # Removed

                path_t1=path_t1, epoch_num=epoch_num, current_sample_global_idx=i,
                output_base_dir=output_base_dir, wandb_run=wandb_run, model_name=model_name
            )
        # --- End VTK and Plotting ---

    # --- Log Probe Error Table to W&B ---
    if wandb_run and probes_enabled and wandb_table_probe_errors_data:
        probe_table_columns = ["Case", "ProbeID", "TargetX", "TargetY", "TargetZ", "ErrorMagnitude"]
        probe_error_table = wandb.Table(columns=probe_table_columns, data=wandb_table_probe_errors_data)
        wandb_run.log({f"{model_name}/Probes/ErrorMagnitudes_Epoch{epoch_num}": probe_error_table}, step=epoch_num, commit=False) # commit=False, main log call will commit

    # --- Aggregation of standard metrics ---
    avg_metrics = {key: float(np.nanmean(values) if len(values) > 0 and np.any(np.isfinite(values)) else np.nan)
                   for key, values in metrics_list.items() if values}

    return_metrics = {
        "val_mse": avg_metrics.get("mse", np.nan),
        "val_rmse_mag": avg_metrics.get("rmse_mag", np.nan),
        "val_mse_div": avg_metrics.get("mse_div", np.nan),
        "val_mse_x": avg_metrics.get("mse_x", np.nan),
        "val_mse_y": avg_metrics.get("mse_y", np.nan),
        "val_mse_z": avg_metrics.get("mse_z", np.nan),
        # "val_mse_vorticity_mag": avg_metrics.get("mse_vorticity_mag", np.nan), # Removed

        "val_cosine_sim": avg_metrics.get("cosine_sim", np.nan),
        "val_avg_max_true_vel_mag": avg_metrics.get("max_true_vel_mag", np.nan),
        "val_avg_max_pred_vel_mag": avg_metrics.get("max_pred_vel_mag", np.nan)
    }
    # wandb_probe_metrics_for_epoch is no longer returned for direct logging of individual metrics
    return return_metrics, probe_data_collected



if __name__ == '__main__':
    from pathlib import Path
    import shutil
    from .utils import set_seed
    from .models import FlowNet  # Example model
    from .data_utils import PairedFrameDataset, make_frame_pairs, create_noisy_dataset_tree
    import meshio  # For creating dummy data

    print("Testing training.py...")
    set_seed(42)
    test_device = get_device("auto")
    print(f"Using device: {test_device}")

    # Create dummy data and dataset
    dummy_data_root = Path("outputs/dummy_train_data_main")
    dummy_noisy_data_root = Path("outputs/dummy_train_noisy_data_main")
    if dummy_data_root.exists(): shutil.rmtree(dummy_data_root)
    if dummy_noisy_data_root.exists(): shutil.rmtree(dummy_noisy_data_root)

    case_cfd = dummy_data_root / "sUbend_train01" / "CFD"
    case_cfd.mkdir(parents=True, exist_ok=True)

    points_np = np.random.rand(30, 3).astype(np.float64)  # More points for kNN
    velocity_np = np.random.rand(30, 3).astype(np.float32)
    dummy_msh = meshio.Mesh(points_np, point_data={"U": velocity_np})

    frame_paths_orig = []
    for i in range(3):  # 3 frames for 2 pairs
        p = case_cfd / f"Frame_{i:02d}_data.vtk"
        meshio.write(str(p), dummy_msh, file_format="vtk")
        frame_paths_orig.append(p)

    # Create noisy version for training
    create_noisy_dataset_tree(dummy_data_root, dummy_noisy_data_root, 0.05, 0.15, overwrite=True)

    train_frame_pairs_noisy = make_frame_pairs(dummy_noisy_data_root)
    assert len(train_frame_pairs_noisy) >= 1, "Not enough frame pairs for training test."

    # Configs
    model_cfg = {"h_dim": 32, "layers": 2}  # Small model for test
    graph_cfg = {"k": 5, "down_n": None, "velocity_key": "U", "noisy_velocity_key_suffix": "_noisy"}
    loss_cfg = {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
    hist_bins = 16

    # Dataset and DataLoader
    train_ds = PairedFrameDataset(
        train_frame_pairs_noisy, graph_cfg, graph_type="knn", use_noisy_data=True, device=test_device
    )
    # PyG DataLoader handles batching of Data objects correctly
    train_loader_pyg = DataLoader(train_ds, batch_size=2, shuffle=True)

    # Model and Optimizer
    test_model = FlowNet(model_cfg).to(test_device)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)

    # Test train_single_epoch
    print("\nTesting train_single_epoch...")
    epoch_metrics = train_single_epoch(
        test_model, train_loader_pyg, optimizer, loss_cfg, hist_bins, test_device
    )
    print(f"Epoch training metrics: {epoch_metrics}")
    assert "total" in epoch_metrics and epoch_metrics["total"] > 0
    print("train_single_epoch test passed.")

    # Test validate_on_pairs
    # Use original (non-noisy) data for this validation example, but point to its paths
    val_pairs_orig = make_frame_pairs(dummy_data_root)
    assert len(val_pairs_orig) >= 1, "Not enough frame pairs for validation test."

    print("\nTesting validate_on_pairs...")
    # For this test, use_noisy_data_for_val = False as we are using original files
    validation_metrics = validate_on_pairs(
        test_model, val_pairs_orig, graph_cfg, use_noisy_data_for_val=False, device=test_device, graph_type="knn"
    )
    print(f"Validation metrics: {validation_metrics}")
    assert "val_mse" in validation_metrics
    print("validate_on_pairs test passed.")

    # Cleanup
    if dummy_data_root.exists(): shutil.rmtree(dummy_data_root)
    if dummy_noisy_data_root.exists(): shutil.rmtree(dummy_noisy_data_root)
    print("\nDummy training test files cleaned up.")
    print("training.py tests complete.")
