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


def train_single_epoch(
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        loss_weights: dict,  # e.g., {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
        histogram_bins: int,
        device: torch.device,
        clip_grad_norm_value: float | None = 1.0
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
        "histogram": 0.0
    }
    num_batches = 0

    for graph_t0, graph_t1 in train_loader:
        graph_t0 = graph_t0.to(device)
        graph_t1 = graph_t1.to(device)  # True velocities are in graph_t1.x

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

        total_loss.backward()

        if clip_grad_norm_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_value)

        optimizer.step()

        epoch_aggregated_losses["total"] += total_loss.item()
        epoch_aggregated_losses["supervised"] += individual_losses["supervised"].item()
        epoch_aggregated_losses["divergence"] += individual_losses["divergence"].item()
        epoch_aggregated_losses["histogram"] += individual_losses["histogram"].item()
        num_batches += 1

    if num_batches > 0:
        for key in epoch_aggregated_losses:
            epoch_aggregated_losses[key] /= num_batches

    return epoch_aggregated_losses


@torch.no_grad()
def validate_on_pairs(
        model: torch.nn.Module,
        val_frame_pairs: list[tuple[Path, Path]],  # List of (path_t0, path_t1)
        graph_config: dict,  # For vtk_to_graph (k_neighbors, downsample_n, keys)
        use_noisy_data_for_val: bool,  # Whether val data itself is noisy
        device: torch.device,
        graph_type: str = "knn",  # "knn" or "full_mesh" for graph construction
        epoch_num: int = -1,  # For naming output files, -1 for non-epoch specific validation
        output_base_dir: str | Path | None = None,  # Base path for saving field VTKs
        save_fields_vtk: bool = False,  # Flag to control saving of VTK files
        wandb_run: wandb.sdk.wandb_run.Run | None = None,  # For logging images
        log_field_image_sample_idx: int = 0,  # Index of the sample in val_frame_pairs to log as an image
        model_name: str = "Model"  # For naming W&B logs
) -> dict:
    """
    Validates the model on a set of paired frames from VTK files.
    Computes MSE, RMSE of velocity magnitude, and MSE of divergence.
    Optionally saves VTK files with true, predicted, and error fields.

    Args:
        model: The trained GNN model.
        val_frame_pairs: List of (path_t0, path_t1) tuples for validation.
        graph_config: Configuration for graph construction. Expects 'k_neighbors' if knn.
        use_noisy_data_for_val: If True, loads noisy versions of validation data.
        device: PyTorch device.
        graph_type: Type of graph to construct for validation ("knn" or "full_mesh").
        epoch_num: Current epoch number, used for naming output directories for VTK fields.
        output_base_dir: Base directory for saving run outputs, where validation_fields will be created.
        save_fields_vtk: If True, saves the VTK files with detailed fields.

    Returns:
        A dictionary with mean validation metrics: "val_mse", "val_rmse_mag", "val_mse_div".
    """
    model.eval()
    metrics_list = {
        "mse": [], "rmse_mag": [], "mse_div": [],
        "mse_x": [], "mse_y": [], "mse_z": [],  # For component-wise MSE
        "mse_vorticity_mag": []  # For MSE of vorticity magnitude
    }

    from .data_utils import vtk_to_knn_graph, vtk_to_fullmesh_graph  # Local import for clarity

    for i, (path_t0, path_t1) in enumerate(val_frame_pairs):
        # For extensive VTK saving, one might save only for a subset of pairs, e.g., if i % N == 0
        # For now, saving for all pairs in the validation set if save_fields_vtk is True.

        if graph_type == "knn":
            # Prepare arguments for vtk_to_knn_graph carefully, mapping 'k' from config
            knn_args = {
                "k_neighbors": graph_config["k"],  # Map 'k' from config to 'k_neighbors'
                "downsample_n": graph_config.get("down_n"),
                "velocity_key": graph_config.get("velocity_key", "U"),
                "noisy_velocity_key_suffix": graph_config.get("noisy_velocity_key_suffix", "_noisy"),
            }
            graph_t0 = vtk_to_knn_graph(
                path_t0,
                **knn_args,  # Pass mapped and other relevant args from graph_config
                use_noisy_data=use_noisy_data_for_val,
                device=device
            )
            graph_t1 = vtk_to_knn_graph(  # Target graph, usually consistent noise setting
                path_t1,
                **knn_args,
                use_noisy_data=use_noisy_data_for_val,
                device=device
            )
        elif graph_type == "full_mesh":
            # For full_mesh, graph_config might contain different keys (e.g. velocity_key, pressure_key)
            graph_t0 = vtk_to_fullmesh_graph(
                path_t0, velocity_key=graph_config.get("velocity_key", "U"),
                pressure_key=graph_config.get("pressure_key", "p"), device=device
            )
            graph_t1 = vtk_to_fullmesh_graph(  # Target graph
                path_t1, velocity_key=graph_config.get("velocity_key", "U"),
                pressure_key=graph_config.get("pressure_key", "p"), device=device
            )
        else:
            raise ValueError(f"Unsupported graph_type for validation: {graph_type}")

        predicted_vel_t1 = model(graph_t0)
        true_vel_t1 = graph_t1.x.to(device)  # Ensure target is on the same device

        # MSE for velocity vectors (overall)
        mse = F.mse_loss(predicted_vel_t1, true_vel_t1).item()
        metrics_list["mse"].append(mse)

        # MSE for each velocity component
        for component_idx, component_label in enumerate(['x', 'y', 'z']):
            mse_comp = F.mse_loss(predicted_vel_t1[:, component_idx], true_vel_t1[:, component_idx]).item()
            metrics_list[f"mse_{component_label}"].append(mse_comp)

        # RMSE for velocity magnitudes
        pred_mag = predicted_vel_t1.norm(dim=1)
        true_mag = true_vel_t1.norm(dim=1)
        rmse_mag = torch.sqrt(F.mse_loss(pred_mag, true_mag)).item()
        metrics_list["rmse_mag"].append(rmse_mag)

        # MSE of divergence (divergence of predicted field vs. divergence of true field)
        # Note: Original code `(cont_div(pred, g0) ** 2).mean().item()` implies target div is 0.
        # Here, we can compare div_pred to div_true if desired, or just penalize non-zero div_pred.
        # For consistency with original, let's use (div_pred^2).mean()
        div_pred = calculate_divergence(predicted_vel_t1, graph_t0)
        # If we want to compare to divergence of true field (requires true_vel_t1 on graph_t0 structure)
        # div_true = calculate_divergence(true_vel_t1, graph_t0) # This might be problematic if meshes differ slightly
        # mse_div = F.mse_loss(div_pred, div_true).item()
        mse_div = (div_pred ** 2).mean().item()  # Penalize non-zero divergence of prediction
        metrics_list["mse_div"].append(mse_div)

        # Attempt to get points_np early if pos data is available.
        points_np = None
        if graph_t1.pos is not None:
            points_np = graph_t1.pos.cpu().numpy()

        # Flag to track if mse_vorticity_mag has been added for this item
        vorticity_metric_added_for_item = False
        error_mag = None # Initialize error_mag to None

        # Save detailed fields to VTK if requested
        if save_fields_vtk and output_base_dir and points_np is not None: # points_np must exist
            try:
                error_mag = torch.norm(predicted_vel_t1 - true_vel_t1, dim=1) # Assigned here if VTK is saved
                frame_name_stem = path_t1.stem
                case_name = path_t1.parent.parent.name
                epoch_folder_name = f"epoch_{epoch_num}" if epoch_num >= 0 else "final_validation"
                vtk_output_dir = Path(output_base_dir) / "validation_fields" / epoch_folder_name / case_name
                vtk_output_dir.mkdir(parents=True, exist_ok=True)
                vtk_file_path = vtk_output_dir / f"{frame_name_stem}_fields.vtk"

                # true_vel_np and pred_vel_np are needed for VTK and vorticity
                true_vel_np = true_vel_t1.cpu().numpy()
                pred_vel_np = predicted_vel_t1.cpu().numpy()

                point_data_for_vtk = {
                    "true_velocity": true_vel_np,
                    "predicted_velocity": pred_vel_np,
                    "velocity_error_magnitude": error_mag.cpu().numpy()
                }

                # Calculate and add vorticity if points are 3D
                if points_np.shape[1] == 3:
                    true_vort_mag = calculate_vorticity_magnitude(points_np, true_vel_np)
                    pred_vort_mag = calculate_vorticity_magnitude(points_np, pred_vel_np)
                    if true_vort_mag is not None and pred_vort_mag is not None:
                        point_data_for_vtk["true_vorticity_magnitude"] = true_vort_mag
                        point_data_for_vtk["predicted_vorticity_magnitude"] = pred_vort_mag

                        # Calculate MSE for vorticity magnitude
                        # Ensure they are torch tensors for mse_loss if needed, or use numpy.
                        # For simplicity, using numpy for mse if arrays are already numpy.
                        if true_vort_mag.shape == pred_vort_mag.shape:  # Basic check
                            mse_vort_mag = np.mean((true_vort_mag - pred_vort_mag) ** 2)
                            metrics_list["mse_vorticity_mag"].append(mse_vort_mag)
                        else:
                            print(
                                f"Warning: Shape mismatch for vorticity magnitudes for {path_t1.name}, cannot compute MSE.")
                    else:
                        # Append a placeholder if vorticity calculation failed for this frame
                        metrics_list["mse_vorticity_mag"].append(np.nan)  # Or some other indicator like -1

                write_vtk_with_fields(
                    filepath=str(vtk_file_path),
                    points=points_np,  # Use points from target graph
                    point_data=point_data_for_vtk
                )
            except Exception as e:
                print(f"Warning: Could not save detailed VTK fields for {path_t1.name}: {e}")
                # Ensure a placeholder for mse_vorticity_mag if VTK saving fails before its calculation for this frame
                if "mse_vorticity_mag" not in metrics_list or len(
                        metrics_list["mse_vorticity_mag"]) <= i:  # Check if already added for this 'i'
                    if points_np.shape[1] == 3:  # Only append if it was supposed to be calculated
                        metrics_list["mse_vorticity_mag"].append(np.nan)
        elif points_np.shape[1] != 3 and "mse_vorticity_mag" not in metrics_list or len(
                metrics_list["mse_vorticity_mag"]) <= i:
            # If data is not 3D, vorticity is not calculated, append NaN for consistency if key exists
            metrics_list.setdefault("mse_vorticity_mag", []).append(np.nan)

        # Log field comparison image to W&B for the specified sample index
        if wandb_run and i == log_field_image_sample_idx and points_np is not None: # Check points_np for plotting
            # Ensure error_mag is defined for plotting.
            # If not calculated during VTK save (because save_fields_vtk=False or other reasons), calculate it now.
            if error_mag is None:
                error_mag = torch.norm(predicted_vel_t1 - true_vel_t1, dim=1)

            try:
                true_vel_mag = true_vel_t1.norm(dim=1).cpu().numpy()
                pred_vel_mag = predicted_vel_t1.norm(dim=1).cpu().numpy()
                # error_mag is now guaranteed to be a tensor if this path is taken.
                err_mag_np = error_mag.cpu().numpy()


                points_plot = points_np # Use points_np which is confirmed not None

                # Create a 2D slice (e.g., points near z=mean(z))
                if points_plot.shape[1] == 3:  # Ensure data is 3D for slicing
                    mean_z = np.mean(points_plot[:, 2])
                    slice_indices = np.where(np.abs(points_plot[:, 2] - mean_z) < 0.05 * (
                                np.max(points_plot[:, 2]) - np.min(points_plot[:, 2]) + 1e-6))[0]
                    if len(slice_indices) < 3:  # Not enough points for triangulation
                        print(f"Warning: Not enough points in slice for sample {i} to generate W&B field image.")
                        slice_points_2d = points_plot[:, :2]  # Fallback to all XY points if slice is too small
                        slice_true_mag = true_vel_mag
                        slice_pred_mag = pred_vel_mag
                        slice_err_mag = err_mag_np
                    else:
                        slice_points_2d = points_plot[slice_indices, :2]
                        slice_true_mag = true_vel_mag[slice_indices]
                        slice_pred_mag = pred_vel_mag[slice_indices]
                        slice_err_mag = err_mag_np[slice_indices]
                else:  # Data is already 2D
                    slice_points_2d = points_plot
                    slice_true_mag = true_vel_mag
                    slice_pred_mag = pred_vel_mag
                    slice_err_mag = err_mag_np

                if len(slice_points_2d) >= 3:  # Need at least 3 points for Delaunay/tricontourf
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    fig.suptitle(f"Epoch {epoch_num} - Sample {i} ({path_t1.stem}) - Z-slice comparison", fontsize=16)

                    try:
                        tri = Delaunay(slice_points_2d)

                        ax = axes[0]
                        contour1 = ax.tricontourf(slice_points_2d[:, 0], slice_points_2d[:, 1], tri.simplices,
                                                  slice_true_mag, levels=14, cmap="jet")
                        fig.colorbar(contour1, ax=ax)
                        ax.set_title("True Velocity Magnitude")
                        ax.set_xlabel("X")
                        ax.set_ylabel("Y")
                        ax.axis('equal')

                        ax = axes[1]
                        contour2 = ax.tricontourf(slice_points_2d[:, 0], slice_points_2d[:, 1], tri.simplices,
                                                  slice_pred_mag, levels=14, cmap="jet")
                        fig.colorbar(contour2, ax=ax)
                        ax.set_title("Predicted Velocity Magnitude")
                        ax.set_xlabel("X")
                        ax.axis('equal')

                        ax = axes[2]
                        contour3 = ax.tricontourf(slice_points_2d[:, 0], slice_points_2d[:, 1], tri.simplices,
                                                  slice_err_mag, levels=14, cmap="Reds")
                        fig.colorbar(contour3, ax=ax)
                        ax.set_title("Error Magnitude")
                        ax.set_xlabel("X")
                        ax.axis('equal')

                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        wandb_run.log({f"{model_name}/Validation_Fields_Epoch{epoch_num}": wandb.Image(buf)})
                        plt.close(fig)
                    except Exception as e_plot:  # Catch errors during plotting (e.g. if all points collinear for Delaunay)
                        print(f"Warning: Failed to generate tricontourf plot for sample {i}: {e_plot}")
                        if 'fig' in locals() and fig is not None: plt.close(fig)  # Ensure figure is closed
                else:
                    print(
                        f"Warning: Not enough unique points in slice for sample {i} ({len(slice_points_2d)} points) to generate W&B field image with tricontourf.")

            except Exception as e_wandb_img:
                print(f"Warning: Could not log W&B field image for sample {i}: {e_wandb_img}")
                if 'fig' in locals() and fig is not None: plt.close(fig)  # Ensure figure is closed

    avg_metrics = {key: float(np.nanmean(values) if np.any(np.isfinite(values)) else 0.0) for key, values in
                   metrics_list.items()}

    # Ensure all expected keys are present in the return, even if empty (though avg_metrics handles this)
    return_metrics = {
        "val_mse": avg_metrics.get("mse", 0.0),
        "val_rmse_mag": avg_metrics.get("rmse_mag", 0.0),
        "val_mse_div": avg_metrics.get("mse_div", 0.0),
        "val_mse_x": avg_metrics.get("mse_x", 0.0),
        "val_mse_y": avg_metrics.get("mse_y", 0.0),
        "val_mse_z": avg_metrics.get("mse_z", 0.0),
        "val_mse_vorticity_mag": avg_metrics.get("mse_vorticity_mag", 0.0)
    }
    return return_metrics


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
