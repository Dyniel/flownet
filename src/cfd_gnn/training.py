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
from torch_geometric.loader import DataLoader # Use PyG DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F # For direct use if needed, though losses module is preferred

from .losses import combined_loss, calculate_divergence # Assuming losses.py is in the same package
from .data_utils import vtk_to_knn_graph # Or a more general graph loader if needed for validation
from .utils import get_device # For device management

def train_single_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_weights: dict, # e.g., {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
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
        graph_t1 = graph_t1.to(device) # True velocities are in graph_t1.x

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
            graph_data=graph_t0, # Divergence and histogram loss use the input graph's structure
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
    val_frame_pairs: list[tuple[Path, Path]], # List of (path_t0, path_t1)
    graph_config: dict, # For vtk_to_graph (k, downsample_n, keys)
    use_noisy_data_for_val: bool, # Whether val data itself is noisy
    device: torch.device,
    graph_type: str = "knn" # "knn" or "full_mesh" for graph construction
) -> dict:
    """
    Validates the model on a set of paired frames from VTK files.
    Computes MSE, RMSE of velocity magnitude, and MSE of divergence.

    Args:
        model: The trained GNN model.
        val_frame_pairs: List of (path_t0, path_t1) tuples for validation.
        graph_config: Configuration for graph construction (k, downsample_n, velocity_key).
        use_noisy_data_for_val: If True, loads noisy versions of validation data.
        device: PyTorch device.
        graph_type: Type of graph to construct for validation ("knn" or "full_mesh").

    Returns:
        A dictionary with mean validation metrics: "val_mse", "val_rmse_mag", "val_mse_div".
    """
    model.eval()
    metrics_list = {"mse": [], "rmse_mag": [], "mse_div": []}

    from .data_utils import vtk_to_knn_graph, vtk_to_fullmesh_graph # Local import for clarity

    for path_t0, path_t1 in val_frame_pairs:
        if graph_type == "knn":
            graph_t0 = vtk_to_knn_graph(
                path_t0, **graph_config, use_noisy_data=use_noisy_data_for_val, device=device
            )
            graph_t1 = vtk_to_knn_graph( # Target graph, usually consistent noise setting
                path_t1, **graph_config, use_noisy_data=use_noisy_data_for_val, device=device
            )
        elif graph_type == "full_mesh":
             # For full_mesh, graph_config might contain different keys (e.g. velocity_key, pressure_key)
            graph_t0 = vtk_to_fullmesh_graph(
                path_t0, velocity_key=graph_config.get("velocity_key", "U"),
                pressure_key=graph_config.get("pressure_key", "p"), device=device
            )
            graph_t1 = vtk_to_fullmesh_graph( # Target graph
                path_t1, velocity_key=graph_config.get("velocity_key", "U"),
                pressure_key=graph_config.get("pressure_key", "p"), device=device
            )
        else:
            raise ValueError(f"Unsupported graph_type for validation: {graph_type}")


        predicted_vel_t1 = model(graph_t0)
        true_vel_t1 = graph_t1.x.to(device) # Ensure target is on the same device

        # MSE for velocity vectors
        mse = F.mse_loss(predicted_vel_t1, true_vel_t1).item()
        metrics_list["mse"].append(mse)

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
        mse_div = (div_pred ** 2).mean().item() # Penalize non-zero divergence of prediction
        metrics_list["mse_div"].append(mse_div)

    avg_metrics = {key: float(np.mean(values)) if values else 0.0 for key, values in metrics_list.items()}
    return {
        "val_mse": avg_metrics["mse"],
        "val_rmse_mag": avg_metrics["rmse_mag"],
        "val_mse_div": avg_metrics["mse_div"]
    }


if __name__ == '__main__':
    from pathlib import Path
    import shutil
    from .utils import set_seed
    from .models import FlowNet # Example model
    from .data_utils import PairedFrameDataset, make_frame_pairs, create_noisy_dataset_tree
    import meshio # For creating dummy data

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

    points_np = np.random.rand(30, 3).astype(np.float64) # More points for kNN
    velocity_np = np.random.rand(30, 3).astype(np.float32)
    dummy_msh = meshio.Mesh(points_np, point_data={"U": velocity_np})

    frame_paths_orig = []
    for i in range(3): # 3 frames for 2 pairs
        p = case_cfd / f"Frame_{i:02d}_data.vtk"
        meshio.write(str(p), dummy_msh, file_format="vtk")
        frame_paths_orig.append(p)

    # Create noisy version for training
    create_noisy_dataset_tree(dummy_data_root, dummy_noisy_data_root, 0.05, 0.15, overwrite=True)

    train_frame_pairs_noisy = make_frame_pairs(dummy_noisy_data_root)
    assert len(train_frame_pairs_noisy) >= 1, "Not enough frame pairs for training test."

    # Configs
    model_cfg = {"h_dim": 32, "layers": 2} # Small model for test
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
