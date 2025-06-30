# -*- coding: utf-8 -*-
"""
utils.py
--------
General utility functions for the CFD GNN project.
"""
import os
import re
import random
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import meshio
import wandb

def set_seed(seed: int, device_specific: bool = True):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_specific and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Potentially add deterministic algorithms, but this can affect performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def sort_frames_by_number(file_list: list[str | Path]) -> list[Path]:
    """
    Sorts a list of file paths (VTK or other) based on a frame number
    extracted from the filename, typically 'Frame_<num>_'.
    Returns a list of Path objects.
    """
    def extract_num(path: Path) -> int:
        # Search for 'Frame_<num>_' and return <num> as int
        m = re.search(r"Frame_(\d+)_", path.name)
        return int(m.group(1)) if m else -1
    return sorted([Path(p) for p in file_list], key=extract_num)

def rand_unit_vector(n_points: int, n_dims: int = 3, dtype=np.float32) -> np.ndarray:
    """Generates n_points random unit vectors of dimension n_dims."""
    v = np.random.normal(size=(n_points, n_dims)).astype(dtype)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12) # Add epsilon for stability
    return v

def write_vtk_with_fields(
    path: str | Path,
    points: np.ndarray,
    point_data: dict[str, np.ndarray],
    cells: dict | None = None,
    file_format: str = "vtk",
    binary: bool = False
):
    """
    Writes a VTK file with specified points and multiple point_data fields.
    - path: Output file path.
    - points: Numpy array of shape [N, 3] for point coordinates (float64 recommended by meshio).
    - point_data: Dictionary where keys are field names (e.g., "velocity", "pressure", "JSD")
                  and values are numpy arrays of shape [N, D] or [N,].
                  D typically 1 for scalar, 3 for vector. Data types should be float32 or similar.
    - cells: Optional. Cell connectivity information (e.g., {"vertex": np.arange(N)[:, None]}).
             If None, defaults to vertex cells for a point cloud.
    - file_format: "vtk" for legacy, "vtu" for XML-based.
    - binary: Whether to write binary VTK/VTU (smaller files).
    """
    if cells is None:
        cells = {"vertex": np.arange(points.shape[0], dtype=np.int64)[:, None]}

    # Ensure point_data values are numpy arrays with appropriate types
    processed_point_data = {}
    for key, arr in point_data.items():
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        # Ensure data is float32 for consistency, unless it's clearly int for JSD or similar
        # This might need adjustment based on typical data types for specific fields.
        if arr.dtype not in [np.float32, np.float64, np.int32, np.int64] or key=="JSD":
             processed_point_data[key] = arr.astype(np.float32)
        else:
            processed_point_data[key] = arr


    mesh = meshio.Mesh(
        points=points.astype(np.float64),  # meshio prefers float64 for points
        cells=cells,
        point_data=processed_point_data
    )
    meshio.write(str(path), mesh, file_format=file_format, binary=binary)
    # print(f"Saved VTK file: {path}")


def load_config(config_path: str | Path | None, default_config: dict | None = None) -> dict:
    """Loads a YAML configuration file, optionally merging with a default config."""
    cfg = default_config.copy() if default_config else {}
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                user_cfg = yaml.safe_load(f)
            if user_cfg:
                cfg.update(user_cfg)
            print(f"Loaded configuration from: {config_path}")
        else:
            print(f"Warning: Config file not found at {config_path}. Using defaults.")
    else:
        print("No config file provided. Using defaults.")
    return cfg

def get_run_name(provided_name: str | None) -> str:
    """Generates a run name with a timestamp if none is provided."""
    if provided_name:
        return provided_name
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def initialize_wandb(config: dict, run_name: str, project_name: str | None = None, entity: str | None = None, output_dir: str | Path = "."):
    """Initializes Weights & Biases for logging."""
    project = project_name or config.get("wandb_project", "cfd_gnn_unspecified")
    entity = entity or config.get("wandb_entity") # Optional

    # Ensure output_dir for wandb files exists
    wandb_dir = Path(output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        dir=str(wandb_dir), # wandb logs its own files here
        # mode="disabled" if config.get("no_wandb") else "online" # if you add a --no-wandb flag
    )
    print(f"W&B run '{run_name}' initialized. Project: {project}, Entity: {entity or 'default'}. Log dir: {wandb_dir}")
    return run

def get_device(requested_device: str = "auto") -> torch.device:
    """Determines the torch device to use."""
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested_device == "cpu":
        return torch.device("cpu")
    # Default to "auto" behavior
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

if __name__ == '__main__':
    # Example usage:
    print("Testing utils.py...")
    set_seed(42)
    print(f"Random number after seed: {random.random()}")

    test_files = ["Frame_10_data.vtk", "Frame_0_data.vtk", "Frame_5_data.vtk"]
    sorted_p = sort_frames_by_number(test_files)
    print(f"Sorted files: {[p.name for p in sorted_p]}")

    unit_vecs = rand_unit_vector(5)
    print(f"Random unit vectors (shape {unit_vecs.shape}):\n{unit_vecs}")

    # Create dummy config for testing
    dummy_cfg_path = Path("dummy_test_config.yaml")
    with open(dummy_cfg_path, 'w') as f:
        yaml.dump({"lr": 0.01, "epochs": 5}, f)

    default_cfg = {"lr": 0.001, "batch_size": 32, "epochs": 10}
    loaded_cfg = load_config(dummy_cfg_path, default_config=default_cfg)
    print(f"Loaded config: {loaded_cfg}")
    loaded_cfg_no_file = load_config(None, default_config=default_cfg)
    print(f"Loaded config (no file): {loaded_cfg_no_file}")
    os.remove(dummy_cfg_path)

    run_name = get_run_name(None)
    print(f"Generated run name: {run_name}")
    run_name_custom = get_run_name("my_test_run")
    print(f"Custom run name: {run_name_custom}")

    device = get_device("auto")
    print(f"Selected device: {device}")

    # Test VTK writing
    test_vtk_path = Path("outputs/test_output.vtk")
    test_vtk_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.random.rand(10, 3)
    velocity = np.random.rand(10, 3)
    pressure = np.random.rand(10)
    jsd_data = np.random.rand(10)
    write_vtk_with_fields(
        test_vtk_path,
        points,
        point_data={"velocity": velocity, "pressure": pressure, "jsd_field": jsd_data}
    )
    print(f"Test VTK written to {test_vtk_path}")
    # To clean up test file: os.remove(test_vtk_path) if test_vtk_path.exists()

    print("utils.py tests complete.")
