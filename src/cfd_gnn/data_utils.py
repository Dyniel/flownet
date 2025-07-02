# -*- coding: utf-8 -*-
"""
data_utils.py
-------------
Functions and classes for data loading, preprocessing, graph construction,
and dataset creation for CFD GNN models.
"""
import os
import glob
from pathlib import Path
import shutil

import numpy as np
import meshio
import torch
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import NearestNeighbors
from itertools import combinations

from .utils import sort_frames_by_number, rand_unit_vector # Assuming utils.py is in the same package

# --------------------------------------------------------------------- #
# Noise Injection
# --------------------------------------------------------------------- #
def inject_noise_to_mesh(mesh: meshio.Mesh, p_min: float, p_max: float,
                         velocity_key: str = "U", position_noise: bool = True) -> meshio.Mesh:
    """
    Injects MRI-like noise to mesh points and a specified point_data field (e.g., velocity).

    Args:
        mesh: The input meshio.Mesh object.
        p_min: Minimum noise percentage.
        p_max: Maximum noise percentage.
        velocity_key: Key for the velocity field in mesh.point_data.
        position_noise: Whether to add noise to point positions.

    Returns:
        A new meshio.Mesh object with added noise.
    """
    pts = mesh.points.astype(np.float32)
    if velocity_key not in mesh.point_data:
        raise KeyError(f"Velocity key '{velocity_key}' not found in mesh point_data.")
    U = mesh.point_data[velocity_key].astype(np.float32)
    N = pts.shape[0]

    # Generate random noise percentages for each point
    noise_pct_array = np.random.uniform(p_min, p_max, (N, 1)).astype(np.float32)

    new_points = pts
    if position_noise:
        # Add noise to positions: pts_n = pts + p * ||pts|| * rand_unit_vector
        # Note: ||pts|| might not be the most physically meaningful scale for positions,
        # consider alternative scaling if this is problematic (e.g., based on bounding box size).
        # For now, sticking to the original script's logic.
        pts_norm = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12 # Add epsilon for stability
        new_points = pts + noise_pct_array * pts_norm * rand_unit_vector(N, pts.shape[1], dtype=np.float32)

    # Add noise to velocity: U_n = U + p * ||U|| * rand_unit_vector
    U_norm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12 # Add epsilon for stability
    U_noisy = U + noise_pct_array * U_norm * rand_unit_vector(N, U.shape[1], dtype=np.float32)

    # Create new point_data dictionary
    new_point_data = mesh.point_data.copy() # Start with original data
    new_point_data[f"{velocity_key}_noisy"] = U_noisy
    new_point_data["noise_percentage"] = (noise_pct_array * 100).reshape(-1) # Store as percentage

    return meshio.Mesh(
        points=new_points.astype(np.float64), # meshio prefers float64 for points
        cells=mesh.cells,
        point_data=new_point_data,
        cell_data=mesh.cell_data, # Preserve cell data if any
        field_data=mesh.field_data # Preserve field data if any
    )

def create_noisy_dataset_tree(
    source_root: str | Path,
    destination_root: str | Path,
    p_min: float,
    p_max: float,
    velocity_key: str = "U",
    position_noise: bool = True,
    overwrite: bool = False
):
    """
    Copies a directory tree of CFD cases and injects noise into VTK files.
    Structure: source_root/case_name/CFD/*.vtk -> destination_root/case_name/CFD/*.vtk

    Args:
        source_root: Path to the root directory of original CFD cases.
        destination_root: Path to the root directory where noisy data will be saved.
        p_min: Minimum noise percentage.
        p_max: Maximum noise percentage.
        velocity_key: Key for velocity data in VTK files.
        position_noise: Whether to add noise to point positions.
        overwrite: If True, overwrites the destination if it exists.
    """
    source_root = Path(source_root)
    destination_root = Path(destination_root)

    if destination_root.exists() and not overwrite:
        print(f"Noisy dataset at {destination_root} already exists. Skipping creation.")
        return
    if destination_root.exists() and overwrite:
        print(f"Overwriting existing noisy dataset at {destination_root}.")
        shutil.rmtree(destination_root)

    destination_root.mkdir(parents=True, exist_ok=True)
    print(f"Creating noisy dataset at: {destination_root}")

    case_dirs = sorted([d for d in source_root.iterdir() if d.is_dir() and d.name.startswith("sUbend")]) # Adapt if case naming changes

    for case_dir in case_dirs:
        src_cfd_dir = case_dir / "CFD"
        if not src_cfd_dir.is_dir():
            print(f"Warning: CFD subdirectory not found in {case_dir}, skipping.")
            continue

        dst_cfd_dir = destination_root / case_dir.name / "CFD"
        dst_cfd_dir.mkdir(parents=True, exist_ok=True)

        vtk_files = sort_frames_by_number(list(src_cfd_dir.glob("*.vtk")))
        if not vtk_files:
            print(f"Warning: No VTK files found in {src_cfd_dir}, skipping case {case_dir.name}.")
            continue

        print(f"  Processing case: {case_dir.name} ({len(vtk_files)} frames)")
        for f_idx, vtk_path in enumerate(vtk_files):
            try:
                mesh = meshio.read(str(vtk_path))
                noisy_mesh = inject_noise_to_mesh(mesh, p_min, p_max, velocity_key, position_noise)

                # Determine output path
                output_vtk_path = dst_cfd_dir / vtk_path.name
                meshio.write(str(output_vtk_path), noisy_mesh, file_format="vtk", binary=False) # Or True for smaller files
                if (f_idx + 1) % 10 == 0 or (f_idx + 1) == len(vtk_files):
                    print(f"    ... processed frame {f_idx+1}/{len(vtk_files)}")

            except Exception as e:
                print(f"Error processing file {vtk_path}: {e}")
    print("Noisy dataset creation complete.")

# --------------------------------------------------------------------- #
# VTK to Graph Conversion
# --------------------------------------------------------------------- #
def vtk_to_knn_graph(
    vtk_path: str | Path,
    k_neighbors: int,
    downsample_n: int | None = None,
    velocity_key: str = "U", # Key for velocity in point_data
    noisy_velocity_key_suffix: str = "_noisy", # Suffix if using noisy data
    use_noisy_data: bool = False, # If true, tries to use "U_noisy" (or similar)
    device: torch.device | str = "cpu"
) -> Data:
    """
    Loads a VTK file, optionally downsamples, builds a k-NN graph,
    and returns a PyTorch Geometric Data object.

    Args:
        vtk_path: Path to the VTK file.
        k_neighbors: Number of neighbors for k-NN.
        downsample_n: Number of points to downsample to. If None or 0, no downsampling.
        velocity_key: The base key for velocity data (e.g., "U").
        noisy_velocity_key_suffix: Suffix appended to velocity_key if use_noisy_data is True.
        use_noisy_data: If True, attempts to load noisy velocity (e.g., "U_noisy").
                        Falls back to base velocity_key if noisy key is not found.
        device: PyTorch device to move tensors to.

    Returns:
        PyTorch Geometric Data object with x, pos, edge_index, edge_attr.
    """
    mesh = meshio.read(str(vtk_path))
    points = mesh.points.astype(np.float32)

    # Check for duplicate points
    unique_points, counts = np.unique(points, axis=0, return_counts=True)
    num_duplicates = points.shape[0] - unique_points.shape[0]
    if num_duplicates > 0:
        print(f"Warning: Found {num_duplicates} duplicate point coordinates in {vtk_path} out of {points.shape[0]} total points. This might affect divergence calculation.")
        # Example: print(f"Top 5 duplicate counts: {counts[counts > 1][:5]}")


    actual_velocity_key = velocity_key
    if use_noisy_data:
        potential_noisy_key = f"{velocity_key}{noisy_velocity_key_suffix}"
        if potential_noisy_key in mesh.point_data:
            actual_velocity_key = potential_noisy_key
        else:
            print(f"Warning: Noisy key '{potential_noisy_key}' not found in {vtk_path}. Using '{velocity_key}'.")

    if actual_velocity_key not in mesh.point_data:
        raise KeyError(f"Velocity key '{actual_velocity_key}' not found in point_data of {vtk_path}.")
    velocities = mesh.point_data[actual_velocity_key].astype(np.float32)

    if points.shape[0] != velocities.shape[0]:
        raise ValueError(f"Mismatch in number of points ({points.shape[0]}) and velocity entries ({velocities.shape[0]}) in {vtk_path}.")

    # Down-sampling
    if downsample_n and points.shape[0] > downsample_n:
        indices = np.linspace(0, points.shape[0] - 1, downsample_n, dtype=int, endpoint=True)
        points = points[indices]
        velocities = velocities[indices]

    num_nodes = points.shape[0]

    # k-NN graph construction
    if num_nodes <= k_neighbors:
        print(f"Warning: Number of nodes ({num_nodes}) is less than or equal to k ({k_neighbors}) in {vtk_path}. Adjusting k.")
        current_k = max(1, num_nodes -1) # At least 1 neighbor if possible
        if current_k == 0 and num_nodes == 1: # Single node graph
            src = np.array([], dtype=np.int64)
            dst = np.array([], dtype=np.int64)
        else:
             nbrs = NearestNeighbors(n_neighbors=current_k + 1, algorithm='auto').fit(points) # +1 because it includes self
             _, indices = nbrs.kneighbors(points)
             src = np.repeat(np.arange(num_nodes), current_k)
             dst = indices[:, 1:].reshape(-1) # Exclude self (first column)
    else:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto').fit(points) # +1 because it includes self
        _, indices = nbrs.kneighbors(points)
        src = np.repeat(np.arange(num_nodes), k_neighbors)
        dst = indices[:, 1:].reshape(-1) # Exclude self (first column)


    edge_index_np = np.vstack([src, dst]).astype(np.int64)

    # Relative positions as edge attributes
    if edge_index_np.shape[1] > 0: # Check if there are any edges
        relative_positions = points[dst] - points[src]
    else: # Handle case with no edges (e.g. single node graph)
        relative_positions = np.empty((0, points.shape[1]), dtype=np.float32)


    return Data(
        x=torch.from_numpy(velocities).to(device),
        pos=torch.from_numpy(points).to(device),
        edge_index=torch.from_numpy(edge_index_np).to(device),
        edge_attr=torch.from_numpy(relative_positions).to(device),
        path=str(vtk_path) # Store path for reference
    )

def vtk_to_fullmesh_graph(
    vtk_path: str | Path,
    velocity_key: str = "U",
    pressure_key: str = "p", # For normalized pressure calculation
    device: torch.device | str = "cpu"
) -> Data:
    """
    Loads a VTK file and builds a graph from its tetrahedral cell connectivity.
    Also calculates normalized pressure and includes it in the Data object.

    Args:
        vtk_path: Path to the VTK file.
        velocity_key: Key for velocity data in point_data.
        pressure_key: Key for pressure data in point_data.
        device: PyTorch device.

    Returns:
        PyTorch Geometric Data object.
    """
    mesh = meshio.read(str(vtk_path))

    points = mesh.points.astype(np.float32)

    # Check for duplicate points
    unique_points_full, counts_full = np.unique(points, axis=0, return_counts=True)
    num_duplicates_full = points.shape[0] - unique_points_full.shape[0]
    if num_duplicates_full > 0:
        print(f"Warning: Found {num_duplicates_full} duplicate point coordinates in {vtk_path} (full_mesh) out of {points.shape[0]} total points.")

    if velocity_key not in mesh.point_data:
        raise KeyError(f"Velocity key '{velocity_key}' not found in {vtk_path}")
    velocities = mesh.point_data[velocity_key].astype(np.float32)

    if pressure_key not in mesh.point_data:
        raise KeyError(f"Pressure key '{pressure_key}' not found in {vtk_path}")
    pressures = mesh.point_data[pressure_key].astype(np.float32)

    num_nodes = points.shape[0]

    # Normalized pressure calculation (as in full_mesh_validation.py)
    inlet_idx = int(np.argmin(points[:, 1]))  # Node with smallest y-coordinate
    p_ref = float(pressures[inlet_idx])
    diff_p = pressures - p_ref
    norm_p_val = float(np.linalg.norm(diff_p))
    if norm_p_val == 0.0: norm_p_val = 1.0 # Avoid division by zero
    normalized_pressure = (pressures - p_ref) / norm_p_val

    # Edge construction from tetrahedral cells
    edges_set = set()
    found_tetra = False
    for cell_block in mesh.cells:
        # meshio cell types can be like "tetra" or "tetra10" (quadratic)
        if cell_block.type.lower().startswith("tetra"):
            found_tetra = True
            # For each tetrahedron, add edges between its vertices
            for tetra_nodes in cell_block.data: # tetra_nodes is an array of 4 node indices
                for i, j in combinations(tetra_nodes, 2):
                    u, v = int(i), int(j)
                    # Add edges lexicographically to avoid duplicates before symmetrizing
                    edges_set.add(tuple(sorted((u,v))))

    if not found_tetra:
        print(f"Warning: No tetrahedral cells found in {vtk_path}. Graph will have no edges based on mesh connectivity.")
        edge_index_np = np.empty((2,0), dtype=np.int64)
        relative_positions = np.empty((0, points.shape[1]), dtype=np.float32)
    else:
        if not edges_set: # Found tetra cells, but they formed no edges (e.g. degenerate mesh)
            print(f"Warning: Tetrahedral cells found in {vtk_path} but resulted in no edges. Graph will be edgeless.")
            edge_index_np = np.empty((2,0), dtype=np.int64)
            relative_positions = np.empty((0, points.shape[1]), dtype=np.float32)
        else:
            # Symmetrize edges
            unique_edges = np.array(list(edges_set), dtype=np.int64) # Shape [num_unique_edges, 2]
            src = np.concatenate([unique_edges[:, 0], unique_edges[:, 1]])
            dst = np.concatenate([unique_edges[:, 1], unique_edges[:, 0]])
            edge_index_np = np.vstack([src, dst]) # Shape [2, 2 * num_unique_edges]
            relative_positions = points[dst] - points[src]


    data = Data(
        x=torch.from_numpy(velocities).to(device),
        pos=torch.from_numpy(points).to(device),
        edge_index=torch.from_numpy(edge_index_np).to(device),
        edge_attr=torch.from_numpy(relative_positions).to(device),
        p_norm=torch.from_numpy(normalized_pressure.astype(np.float32)).to(device),
        inlet_idx=torch.tensor(inlet_idx, device=device, dtype=torch.long),
        path=str(vtk_path)
    )
    return data

# --------------------------------------------------------------------- #
# Dataset Class for Paired Frames
# --------------------------------------------------------------------- #
def make_frame_pairs(data_root: str | Path, case_glob: str = "sUbend*") -> list[tuple[Path, Path]]:
    """
    Scans a directory for CFD cases and creates pairs of consecutive VTK frames.
    Assumes structure: data_root/case_name/CFD/*.vtk

    Args:
        data_root: Root directory containing case subdirectories.
        case_glob: Glob pattern for case directories.

    Returns:
        List of (frame_t0_path, frame_t1_path) tuples.
    """
    data_root = Path(data_root)
    frame_pairs = []

    case_dirs = sorted([d for d in data_root.glob(case_glob) if d.is_dir()])

    for case_dir in case_dirs:
        cfd_dir = case_dir / "CFD"
        if not cfd_dir.is_dir():
            print(f"Warning: CFD subdirectory not found in {case_dir}, skipping.")
            continue

        vtk_files = sort_frames_by_number(list(cfd_dir.glob("*.vtk")))
        if len(vtk_files) < 2:
            print(f"Warning: Less than 2 VTK files in {cfd_dir} for case {case_dir.name}, cannot make pairs.")
            continue

        # Create pairs (frame_i, frame_i+1)
        for i in range(len(vtk_files) - 1):
            frame_pairs.append((vtk_files[i], vtk_files[i+1]))

    if not frame_pairs:
        print(f"Warning: No frame pairs were created from {data_root} with pattern {case_glob}.")
    return frame_pairs


class PairedFrameDataset(Dataset):
    """
    PyTorch Geometric Dataset for pairs of consecutive CFD frames.
    Each item consists of two graphs (graph_t0, graph_t1).
    """
    def __init__(self, frame_pairs: list[tuple[Path, Path]], graph_config: dict,
                 graph_type: str = "knn", # "knn" or "full_mesh"
                 use_noisy_data: bool = False, device: str | torch.device = "cpu"):
        super().__init__()
        self.frame_pairs = frame_pairs
        self.graph_config = graph_config # Contains k, downsample_n, velocity_key etc.
        self.use_noisy_data = use_noisy_data
        self.device = device
        self.graph_type = graph_type

        if not self.frame_pairs:
            raise ValueError("frame_pairs list cannot be empty.")

    def len(self) -> int:
        return len(self.frame_pairs)

    def get(self, idx: int) -> tuple[Data, Data]:
        path_t0, path_t1 = self.frame_pairs[idx]

        if self.graph_type == "knn":
            graph_t0 = vtk_to_knn_graph(
                path_t0,
                k_neighbors=self.graph_config["k"],
                downsample_n=self.graph_config.get("down_n"), # Optional
                velocity_key=self.graph_config.get("velocity_key", "U"),
                noisy_velocity_key_suffix=self.graph_config.get("noisy_velocity_key_suffix", "_noisy"),
                use_noisy_data=self.use_noisy_data,
                device=self.device
            )
            graph_t1 = vtk_to_knn_graph(
                path_t1,
                k_neighbors=self.graph_config["k"],
                downsample_n=self.graph_config.get("down_n"),
                velocity_key=self.graph_config.get("velocity_key", "U"),
                noisy_velocity_key_suffix=self.graph_config.get("noisy_velocity_key_suffix", "_noisy"),
                use_noisy_data=self.use_noisy_data, # Typically target is also noisy if input is
                device=self.device
            )
        elif self.graph_type == "full_mesh":
            # For full_mesh, downsampling and k are not applicable from graph_config in the same way
            # Noisy data handling for full_mesh might need specific keys if different from kNN.
            # Assuming velocity_key and pressure_key are relevant.
            graph_t0 = vtk_to_fullmesh_graph(
                path_t0,
                velocity_key=self.graph_config.get("velocity_key", "U"),
                pressure_key=self.graph_config.get("pressure_key", "p"),
                device=self.device
            )
            graph_t1 = vtk_to_fullmesh_graph(
                path_t1,
                velocity_key=self.graph_config.get("velocity_key", "U"),
                pressure_key=self.graph_config.get("pressure_key", "p"),
                device=self.device
            )
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

        return graph_t0, graph_t1


if __name__ == '__main__':
    print("Testing data_utils.py...")
    # Create dummy VTK files and directory structure for testing
    test_data_root = Path("outputs/dummy_test_data")
    test_noisy_root = Path("outputs/dummy_test_noisy_data")

    if test_data_root.exists(): shutil.rmtree(test_data_root)
    if test_noisy_root.exists(): shutil.rmtree(test_noisy_root)

    case1_cfd = test_data_root / "sUbend_001" / "CFD"
    case1_cfd.mkdir(parents=True, exist_ok=True)

    # Create a simple mesh for testing
    points_np = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,1]], dtype=np.float64)
    velocity_np = np.random.rand(5, 3).astype(np.float32)
    pressure_np = np.random.rand(5).astype(np.float32)
    # Minimal cells for full_mesh_graph (one tetra) - needs at least 4 points for a tetra
    # For kNN, cells are not strictly needed as it's based on point proximity
    cells_dict = [meshio.CellBlock("tetra", np.array([[0,1,2,4]]))] if points_np.shape[0] >=4 else None


    dummy_mesh = meshio.Mesh(points_np, cells_dict,
                             point_data={"U": velocity_np, "p": pressure_np})

    frame0_path = case1_cfd / "Frame_00_data.vtk"
    frame1_path = case1_cfd / "Frame_01_data.vtk"
    meshio.write(str(frame0_path), dummy_mesh, file_format="vtk")
    meshio.write(str(frame1_path), dummy_mesh, file_format="vtk") # Write same mesh for simplicity

    # 1. Test Noise Injection
    print("\nTesting Noise Injection...")
    create_noisy_dataset_tree(test_data_root, test_noisy_root, p_min=0.05, p_max=0.15, overwrite=True)
    noisy_frame0_path = test_noisy_root / "sUbend_001" / "CFD" / "Frame_00_data.vtk"
    assert noisy_frame0_path.exists(), "Noisy file not created."
    noisy_mesh_read = meshio.read(noisy_frame0_path)
    assert "U_noisy" in noisy_mesh_read.point_data, "U_noisy field missing."
    assert "noise_percentage" in noisy_mesh_read.point_data, "noise_percentage field missing."
    print("Noise injection test passed.")

    # 2. Test k-NN Graph Conversion
    print("\nTesting k-NN Graph Conversion...")
    graph_config_knn = {"k": 2, "down_n": None, "velocity_key": "U"} # k < N-1
    if points_np.shape[0] <= graph_config_knn["k"]: # Adjust k if too large for dummy data
        graph_config_knn["k"] = max(1, points_np.shape[0]-2)

    knn_graph = vtk_to_knn_graph(frame0_path, **graph_config_knn, use_noisy_data=False, device="cpu")
    print(f"k-NN graph (original data): {knn_graph}")
    assert knn_graph.x.shape[0] == points_np.shape[0]
    assert knn_graph.edge_index.shape[0] == 2

    noisy_knn_graph = vtk_to_knn_graph(noisy_frame0_path, **graph_config_knn, use_noisy_data=True, device="cpu")
    print(f"k-NN graph (noisy data): {noisy_knn_graph}")
    assert noisy_knn_graph.x.shape[0] == points_np.shape[0]
    print("k-NN graph conversion test passed.")

    # 3. Test Full Mesh Graph Conversion (if cells were defined)
    if cells_dict:
        print("\nTesting Full Mesh Graph Conversion...")
        full_mesh_graph = vtk_to_fullmesh_graph(frame0_path, velocity_key="U", pressure_key="p", device="cpu")
        print(f"Full mesh graph: {full_mesh_graph}")
        assert full_mesh_graph.x.shape[0] == points_np.shape[0]
        assert "p_norm" in full_mesh_graph
        # Number of edges depends on unique edges in tetra: 1 tetra = 6 unique edges * 2 (symmetric) = 12
        if points_np.shape[0] >= 4:
             assert full_mesh_graph.edge_index.shape[1] >= 6*2 # 6 unique edges from one tetra, made symmetric
        print("Full mesh graph conversion test passed.")
    else:
        print("\nSkipping Full Mesh Graph Conversion test (not enough points for tetra or no cells).")


    # 4. Test Frame Pairing and Dataset
    print("\nTesting Frame Pairing and Dataset...")
    frame_pairs = make_frame_pairs(test_data_root)
    assert len(frame_pairs) == 1, f"Expected 1 pair, got {len(frame_pairs)}"
    assert frame_pairs[0] == (frame0_path, frame1_path)

    paired_ds_knn = PairedFrameDataset(frame_pairs, graph_config_knn, graph_type="knn", device="cpu")
    g0_knn, g1_knn = paired_ds_knn[0]
    print(f"Paired k-NN graphs: G0={g0_knn}, G1={g1_knn}")
    assert g0_knn.x.shape[0] == points_np.shape[0]

    if cells_dict: # Only test full_mesh dataset if we could build full_mesh graphs
        graph_config_full = {"velocity_key": "U", "pressure_key": "p"}
        paired_ds_full = PairedFrameDataset(frame_pairs, graph_config_full, graph_type="full_mesh", device="cpu")
        g0_full, g1_full = paired_ds_full[0]
        print(f"Paired full mesh graphs: G0={g0_full}, G1={g1_full}")
        assert g0_full.x.shape[0] == points_np.shape[0]

    print("Frame pairing and dataset tests passed.")

    # Clean up dummy files
    if test_data_root.exists(): shutil.rmtree(test_data_root)
    if test_noisy_root.exists(): shutil.rmtree(test_noisy_root)
    print("\nDummy test files and directories cleaned up.")
    print("data_utils.py tests complete.")
