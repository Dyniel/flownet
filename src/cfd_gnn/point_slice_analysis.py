# -*- coding: utf-8 -*-
"""
point_slice_analysis.py
-----------------------
Functions for extracting data at specific points or along slices from CFD simulation data.
"""
import numpy as np
from scipy.spatial import KDTree
import warnings

def get_points_data(
    mesh_points_coords: np.ndarray,
    target_query_points: list[list[float]] | np.ndarray,
    mesh_velocity_data: np.ndarray,
    time_value: float,
    velocity_field_name: str = "velocity"
) -> list[dict]:
    """
    Extracts data (e.g., velocity) at specified target query points from mesh data using nearest-neighbor lookup.

    Args:
        mesh_points_coords: NumPy array of mesh point coordinates, shape [num_mesh_points, 3].
        target_query_points: List or NumPy array of target XYZ coordinates to query, shape [num_target_points, 3].
        mesh_velocity_data: NumPy array of velocity vectors at each mesh point, shape [num_mesh_points, 3].
        time_value: The simulation time corresponding to this data.
        velocity_field_name: Name of the velocity field (for output dict key).

    Returns:
        A list of dictionaries. Each dictionary contains data for one target query point:
        {
            'time': float,
            'target_point_coords': list[float], # The [x,y,z] of the query point
            'actual_point_coords': list[float], # The [x,y,z] of the nearest mesh point
            'distance_to_actual': float,        # Distance between target and actual mesh point
            'velocity_field_name': list[float]  # e.g. 'velocity': [vx,vy,vz] at the actual point
        }
        Returns an empty list if mesh_points_coords is empty or target_query_points is empty.
    """
    if not isinstance(mesh_points_coords, np.ndarray) or mesh_points_coords.ndim != 2 or mesh_points_coords.shape[1] != 3:
        raise ValueError("mesh_points_coords must be a NumPy array of shape [N, 3].")
    if not isinstance(mesh_velocity_data, np.ndarray) or mesh_velocity_data.ndim != 2 or mesh_velocity_data.shape[1] != 3:
        raise ValueError("mesh_velocity_data must be a NumPy array of shape [N, 3].")
    if mesh_points_coords.shape[0] != mesh_velocity_data.shape[0]:
        raise ValueError("mesh_points_coords and mesh_velocity_data must have the same number of points (first dimension).")

    if mesh_points_coords.shape[0] == 0:
        warnings.warn("mesh_points_coords is empty. Cannot extract point data.", UserWarning)
        return []

    target_query_points_np = np.array(target_query_points, dtype=float)
    if target_query_points_np.ndim != 2 or target_query_points_np.shape[1] != 3:
        if target_query_points_np.size == 0: # Empty list of points passed
             warnings.warn("target_query_points is empty. No data will be extracted.", UserWarning)
             return []
        raise ValueError("target_query_points must be a list or array of shape [M, 3].")

    if target_query_points_np.shape[0] == 0: # Also handles empty list case if it made it here
        return []

    # Build KDTree for efficient nearest neighbor search on the mesh points
    try:
        kdtree = KDTree(mesh_points_coords)
    except Exception as e:
        # KDTree can fail for various reasons, e.g. all points collinear in 2D for 3D tree
        warnings.warn(f"Failed to build KDTree from mesh_points_coords: {e}. Cannot extract point data.", UserWarning)
        return []


    # Query the tree for nearest neighbors to the target_query_points
    # distances will be an array of shape [num_target_points]
    # indices will be an array of shape [num_target_points]
    distances, indices = kdtree.query(target_query_points_np, k=1)

    results = []
    for i, target_point in enumerate(target_query_points_np):
        mesh_point_idx = indices[i]
        actual_mesh_point_coords = mesh_points_coords[mesh_point_idx]
        velocity_at_mesh_point = mesh_velocity_data[mesh_point_idx]
        distance_to_mesh_point = distances[i]

        results.append({
            'time': time_value,
            'target_point_coords': target_point.tolist(),
            'actual_point_coords': actual_mesh_point_coords.tolist(),
            'distance_to_actual': float(distance_to_mesh_point),
            velocity_field_name: velocity_at_mesh_point.tolist()
        })

    return results

if __name__ == '__main__':
    print("Testing point_slice_analysis.py: get_points_data...")

    # Sample mesh data
    mesh_pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]
    ])
    mesh_vel = np.random.rand(8, 3).astype(np.float32)

    # Target query points
    query_pts = [
        [0.1, 0.1, 0.1], # Near [0,0,0]
        [0.9, 0.1, 0.1], # Near [1,0,0]
        [0.5, 0.5, 0.5]  # Equidistant from many
    ]

    time_val_test = 1.234

    extracted_data = get_points_data(mesh_pts, query_pts, mesh_vel, time_val_test)

    assert len(extracted_data) == len(query_pts)
    print(f"Extracted data for {len(extracted_data)} points at time {time_val_test}:")
    for i, data_dict in enumerate(extracted_data):
        print(f"  Query Point {i+1}: {data_dict['target_point_coords']}")
        print(f"    Actual Mesh Point: {data_dict['actual_point_coords']}")
        print(f"    Distance: {data_dict['distance_to_actual']:.4f}")
        print(f"    Velocity: {data_dict['velocity']}")
        assert data_dict['time'] == time_val_test
        assert len(data_dict['velocity']) == 3

    # Test with empty query points
    extracted_empty_query = get_points_data(mesh_pts, [], mesh_vel, time_val_test)
    assert len(extracted_empty_query) == 0
    print("\nTest with empty query_pts list passed.")

    # Test with empty mesh points
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extracted_empty_mesh = get_points_data(np.empty((0,3)), query_pts, np.empty((0,3)), time_val_test)
        assert len(extracted_empty_mesh) == 0
        assert len(w) == 1 and "mesh_points_coords is empty" in str(w[-1].message)
    print("Test with empty mesh_pts list passed (warning issued).")

    print("\nget_points_data tests completed.")


def get_slice_data(
    mesh_points_coords: np.ndarray,
    mesh_velocity_data: np.ndarray,
    time_value: float,
    slice_axis_idx: int,      # 0 for X, 1 for Y, 2 for Z
    slice_position: float,    # Coordinate value for the slice center
    slice_thickness: float,   # Thickness of the slice
    velocity_field_name: str = "velocity" # Name for the velocity field in any output dicts
) -> dict:
    """
    Extracts data for points falling within a defined slice.

    Args:
        mesh_points_coords: NumPy array of mesh point coordinates, shape [N, 3].
        mesh_velocity_data: NumPy array of velocity vectors at each mesh point, shape [N, 3].
        time_value: The simulation time corresponding to this data.
        slice_axis_idx: Index of the axis to slice along (0 for X, 1 for Y, 2 for Z).
        slice_position: Coordinate value for the center of the slice.
        slice_thickness: Total thickness of the slice.
        velocity_field_name: Name for the velocity field (not directly used in this basic version's output dict
                             but good for consistency if extended).

    Returns:
        A dictionary containing data for the slice:
        {
            'time': float,
            'slice_axis_idx': int,
            'slice_position': float,
            'slice_thickness': float,
            'num_points_in_slice': int,
            'avg_velocity_magnitude': float (np.nan if no points in slice),
            'points_in_slice': np.ndarray (shape [M, 3], M points in slice),
            'velocities_in_slice': np.ndarray (shape [M, 3], velocities of M points)
        }
        Returns a dict with num_points_in_slice=0 and NaN metrics if no points are found.
    """
    if not isinstance(mesh_points_coords, np.ndarray) or mesh_points_coords.ndim != 2 or mesh_points_coords.shape[1] != 3:
        raise ValueError("mesh_points_coords must be a NumPy array of shape [N, 3].")
    if not isinstance(mesh_velocity_data, np.ndarray) or mesh_velocity_data.ndim != 2 or mesh_velocity_data.shape[1] != 3:
        raise ValueError("mesh_velocity_data must be a NumPy array of shape [N, 3].")
    if mesh_points_coords.shape[0] != mesh_velocity_data.shape[0]:
        raise ValueError("mesh_points_coords and mesh_velocity_data must have the same number of points.")

    if not (0 <= slice_axis_idx <= 2):
        raise ValueError("slice_axis_idx must be 0, 1, or 2.")
    if slice_thickness < 0:
        # Allow zero thickness for an "exact" plane, though floating point makes this tricky.
        # Smallest positive float might be better for "thin" in practice.
        warnings.warn("slice_thickness is negative. Using abs(slice_thickness).", UserWarning)
        slice_thickness = abs(slice_thickness)

    half_thickness = slice_thickness / 2.0
    lower_bound = slice_position - half_thickness
    upper_bound = slice_position + half_thickness

    point_coords_on_axis = mesh_points_coords[:, slice_axis_idx]

    # Get indices of points within the slice bounds
    slice_indices = np.where(
        (point_coords_on_axis >= lower_bound) & (point_coords_on_axis <= upper_bound)
    )[0]

    num_points_in_slice = len(slice_indices)
    avg_vel_mag = np.nan
    points_in_slice_coords = np.empty((0, 3), dtype=mesh_points_coords.dtype)
    velocities_in_slice_data = np.empty((0, 3), dtype=mesh_velocity_data.dtype)

    if num_points_in_slice > 0:
        points_in_slice_coords = mesh_points_coords[slice_indices]
        velocities_in_slice_data = mesh_velocity_data[slice_indices]

        velocity_magnitudes_in_slice = np.linalg.norm(velocities_in_slice_data, axis=1)
        avg_vel_mag = float(np.mean(velocity_magnitudes_in_slice))
    else:
        warnings.warn(
            f"No points found in slice: axis={slice_axis_idx}, pos={slice_position:.3f}, thick={slice_thickness:.3f}",
            UserWarning
        )

    return {
        'time': time_value,
        'slice_axis_idx': slice_axis_idx,
        'slice_position': slice_position,
        'slice_thickness': slice_thickness,
        'num_points_in_slice': num_points_in_slice,
        'avg_velocity_magnitude': avg_vel_mag,
        'points_in_slice': points_in_slice_coords,
        'velocities_in_slice': velocities_in_slice_data
    }


if __name__ == '__main__':
    print("Testing point_slice_analysis.py: get_points_data...")

    # Sample mesh data
    mesh_pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        # Add points for slice test
        [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
        [0.5, 0.2, 0.2], [0.5, 0.8, 0.8] # These should be in an X=0.5 slice
    ])
    mesh_vel = np.random.rand(mesh_pts.shape[0], 3).astype(np.float32)

    # Target query points
    query_pts = [
        [0.1, 0.1, 0.1], # Near [0,0,0]
        [0.9, 0.1, 0.1], # Near [1,0,0]
        [0.5, 0.5, 0.5]  # Equidistant from many
    ]

    time_val_test = 1.234

    extracted_data = get_points_data(mesh_pts, query_pts, mesh_vel, time_val_test)

    assert len(extracted_data) == len(query_pts)
    print(f"Extracted data for {len(extracted_data)} points at time {time_val_test}:")
    for i, data_dict in enumerate(extracted_data):
        print(f"  Query Point {i+1}: {data_dict['target_point_coords']}")
        print(f"    Actual Mesh Point: {data_dict['actual_point_coords']}")
        print(f"    Distance: {data_dict['distance_to_actual']:.4f}")
        print(f"    Velocity: {data_dict['velocity']}")
        assert data_dict['time'] == time_val_test
        assert len(data_dict['velocity']) == 3

    # Test with empty query points
    extracted_empty_query = get_points_data(mesh_pts, [], mesh_vel, time_val_test)
    assert len(extracted_empty_query) == 0
    print("\nTest with empty query_pts list passed.")

    # Test with empty mesh points
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extracted_empty_mesh = get_points_data(np.empty((0,3)), query_pts, np.empty((0,3)), time_val_test)
        assert len(extracted_empty_mesh) == 0
        assert len(w) >= 1 # Expect at least one warning
        # Check if the specific warning is present
        assert any("mesh_points_coords is empty" in str(warn_item.message) for warn_item in w)
    print("Test with empty mesh_pts list passed (warning issued).")

    print("\nget_points_data tests completed.")

    print("\n--- Testing get_slice_data ---")
    # Slice along X axis (axis_idx=0) at X=0.5 with thickness 0.1
    # Points [0.5,0.2,0.2] and [0.5,0.8,0.8] should be included.
    # Also [0.5,0.0,0.5], [0.5,1.0,0.5], [0.5,0.5,0.0], [0.5,0.5,1.0]
    slice_def_x = {
        "slice_axis_idx": 0,
        "slice_position": 0.5,
        "slice_thickness": 0.1 # X values between 0.45 and 0.55
    }
    slice_info_x = get_slice_data(mesh_pts, mesh_vel, time_val_test, **slice_def_x)

    print(f"Slice X data: {slice_info_x['num_points_in_slice']} points, avg_vel_mag={slice_info_x['avg_velocity_magnitude']:.4f}")
    assert slice_info_x['time'] == time_val_test
    assert slice_info_x['slice_axis_idx'] == 0
    assert slice_info_x['num_points_in_slice'] == 6 # Based on sample data: (0.5,y,z) points
    assert not np.isnan(slice_info_x['avg_velocity_magnitude'])
    assert slice_info_x['points_in_slice'].shape == (6, 3)
    assert slice_info_x['velocities_in_slice'].shape == (6, 3)

    # Test an empty slice (Y=10.0, far from data)
    slice_def_y_empty = {
        "slice_axis_idx": 1,
        "slice_position": 10.0,
        "slice_thickness": 0.1
    }
    with warnings.catch_warnings(record=True) as w_empty_slice:
        warnings.simplefilter("always")
        slice_info_y_empty = get_slice_data(mesh_pts, mesh_vel, time_val_test, **slice_def_y_empty)
        assert len(w_empty_slice) == 1 and "No points found in slice" in str(w_empty_slice[-1].message)

    print(f"Slice Y (empty) data: {slice_info_y_empty['num_points_in_slice']} points, avg_vel_mag={slice_info_y_empty['avg_velocity_magnitude']:.4f}")
    assert slice_info_y_empty['num_points_in_slice'] == 0
    assert np.isnan(slice_info_y_empty['avg_velocity_magnitude'])
    assert slice_info_y_empty['points_in_slice'].shape == (0,3)
    assert slice_info_y_empty['velocities_in_slice'].shape == (0,3)

    print("get_slice_data tests completed.")
