# -*- coding: utf-8 -*-
"""
metrics.py
----------
Evaluation metrics for CFD GNN models, including physical metrics like TKE,
cosine similarity, and Jensen-Shannon Divergence for velocity histograms.
"""
import numpy as np
import torch

# Small epsilon value to prevent division by zero or log(0)
EPS = 1e-9


def turbulent_kinetic_energy(velocity_field: np.ndarray | torch.Tensor) -> float:
    """
    Calculates the mean Turbulent Kinetic Energy (TKE) per unit mass.
    TKE = 0.5 * mean(u'^2 + v'^2 + w'^2)
    If the input is instantaneous velocity U, it calculates 0.5 * mean(U·U).

    Args:
        velocity_field: A numpy array or torch tensor of shape [num_points, 3] or [batch, num_points, 3]
                        representing velocity vectors (u, v, w).

    Returns:
        Mean TKE value (scalar float).
    """
    if isinstance(velocity_field, torch.Tensor):
        # Square of velocity magnitudes: (U·U) = U_x^2 + U_y^2 + U_z^2
        # This is norm(dim=1)^2 if velocity_field is [N,3]
        # Or norm(dim=-1)^2 if velocity_field is [B,N,3]
        squared_magnitudes = torch.sum(velocity_field ** 2, dim=-1)
        tke_val = 0.5 * torch.mean(squared_magnitudes)
        return float(tke_val.item())
    elif isinstance(velocity_field, np.ndarray):
        squared_magnitudes = np.sum(velocity_field ** 2, axis=-1)
        tke_val = 0.5 * np.mean(squared_magnitudes)
        return float(tke_val)
    else:
        raise TypeError("Input velocity_field must be a NumPy array or PyTorch tensor.")


def cosine_similarity_metric(
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        reduction: str = 'mean'  # 'mean' or 'none'
) -> float | np.ndarray | torch.Tensor:
    """
    Computes the cosine similarity between predicted and target velocity fields.

    Args:
        predictions: Predicted velocity vectors, shape [num_points, 3] or [batch, num_points, 3].
        targets: Target velocity vectors, same shape as predictions.
        reduction: 'mean' to return average similarity, 'none' to return per-point similarity.

    Returns:
        Scalar float for mean cosine similarity, or tensor/array if reduction is 'none'.
    """
    is_torch = isinstance(predictions, torch.Tensor)

    if is_torch:
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets).to(predictions.device).type_as(predictions)

        dot_product = torch.sum(predictions * targets, dim=-1)
        norm_pred = torch.linalg.norm(predictions, dim=-1)
        norm_target = torch.linalg.norm(targets, dim=-1)

        similarity = dot_product / (norm_pred * norm_target + EPS)

        if reduction == 'mean':
            return float(torch.mean(similarity).item())
        elif reduction == 'none':
            return similarity
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    else:  # Numpy
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets, dtype=predictions.dtype)

        dot_product = np.sum(predictions * targets, axis=-1)
        norm_pred = np.linalg.norm(predictions, axis=-1)
        norm_target = np.linalg.norm(targets, axis=-1)

        similarity = dot_product / (norm_pred * norm_target + EPS)

        if reduction == 'mean':
            return float(np.mean(similarity))
        elif reduction == 'none':
            return similarity
        else:
            raise ValueError(f"Unknown reduction: {reduction}")


def compute_jsd_histograms(
        real_magnitudes_time_series: np.ndarray,
        pred_magnitudes_time_series: np.ndarray,
        num_bins: int,
        min_max_range: tuple[float, float] | None = None,  # Optional: precomputed (min, max) for histogram bins
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Jensen-Shannon Divergence (JSD) per point based on histograms
    of velocity magnitudes over a time series.

    Args:
        real_magnitudes_time_series: NumPy array of shape [num_time_steps, num_points]
                                     containing real velocity magnitudes.
        pred_magnitudes_time_series: NumPy array of shape [num_time_steps, num_points]
                                     containing predicted velocity magnitudes.
        num_bins: Number of bins for histogram construction.
        min_max_range: Optional tuple (min_val, max_val) to define histogram range.
                       If None, it's computed from the data.

    Returns:
        A tuple containing:
            - jsd_per_point: NumPy array of shape [num_points] with JSD values.
            - histograms_real: NumPy array of shape [num_points, num_bins] with real PDFs.
            - histograms_pred: NumPy array of shape [num_points, num_bins] with predicted PDFs.
    """
    if real_magnitudes_time_series.shape != pred_magnitudes_time_series.shape:
        raise ValueError("Shape mismatch between real and predicted magnitude time series.")
    if real_magnitudes_time_series.ndim != 2:
        raise ValueError("Magnitude time series should be 2D [num_time_steps, num_points].")

    num_time_steps, num_points = real_magnitudes_time_series.shape

    # Determine histogram range (globally for all points)
    if min_max_range:
        min_val, max_val = min_max_range
    else:
        min_val = min(real_magnitudes_time_series.min(), pred_magnitudes_time_series.min())
        max_val = max(real_magnitudes_time_series.max(), pred_magnitudes_time_series.max())

    # Ensure range is valid
    if min_val >= max_val:
        max_val = min_val + EPS  # Add epsilon if min and max are too close or equal

    bin_edges = np.linspace(min_val, max_val, num_bins + 1, dtype=np.float64)

    jsd_per_point = np.zeros(num_points, dtype=np.float64)
    pdfs_real = np.zeros((num_points, num_bins), dtype=np.float64)
    pdfs_pred = np.zeros((num_points, num_bins), dtype=np.float64)

    for i in range(num_points):
        # Create histograms for the i-th point over time
        hist_real_counts, _ = np.histogram(real_magnitudes_time_series[:, i], bins=bin_edges)
        hist_pred_counts, _ = np.histogram(pred_magnitudes_time_series[:, i], bins=bin_edges)

        # Convert counts to probabilities (PDFs), adding EPS for stability
        p_dist = hist_real_counts.astype(np.float64) + EPS
        q_dist = hist_pred_counts.astype(np.float64) + EPS

        p_dist /= p_dist.sum()
        q_dist /= q_dist.sum()

        pdfs_real[i, :] = p_dist
        pdfs_pred[i, :] = q_dist

        # Compute JSD
        m_dist = 0.5 * (p_dist + q_dist)

        # Kullback-Leibler divergences
        kl_pm = np.sum(p_dist * np.log(p_dist / m_dist))  # log base e
        kl_qm = np.sum(q_dist * np.log(q_dist / m_dist))

        jsd_per_point[i] = 0.5 * (kl_pm + kl_qm)

    return jsd_per_point, pdfs_real, pdfs_pred


# --- Vorticity Calculation (using PyVista) ---
def _create_pyvista_grid(points_np: np.ndarray, velocity_np: np.ndarray | None = None):
    """
    Creates a PyVista PolyData object from points and (optionally) velocity data.
    Handles 2D or 3D points/velocities by ensuring they are 3D for PyVista.
    """
    import pyvista as pv  # Local import

    if points_np.ndim != 2 or points_np.shape[1] < 2 or points_np.shape[1] > 3:
        raise ValueError(f"Points must be [N,2] or [N,3], got shape {points_np.shape}")

    points_pv = np.zeros((points_np.shape[0], 3), dtype=points_np.dtype)
    points_pv[:, :points_np.shape[1]] = points_np

    grid = pv.PolyData(points_pv)

    if velocity_np is not None:
        if velocity_np.shape[0] != points_np.shape[0]:
            raise ValueError("Points and velocity arrays must have the same number of points.")
        if velocity_np.ndim != 2 or velocity_np.shape[1] < 2 or velocity_np.shape[1] > 3:
            raise ValueError(f"Velocity must be [N,2] or [N,3], got shape {velocity_np.shape}")

        velocity_pv = np.zeros((velocity_np.shape[0], 3), dtype=velocity_np.dtype)
        velocity_pv[:, :velocity_np.shape[1]] = velocity_np
        grid["velocity"] = velocity_pv
        grid.active_vectors_name = "velocity"  # Set active vector for filters like 'derivatives'

    return grid


def calculate_vorticity_magnitude(points_np: np.ndarray, velocity_np: np.ndarray) -> np.ndarray:
    """
    Calculates the magnitude of vorticity for a given velocity field on a set of points.
    Uses PyVista for the underlying computation.

    Args:
        points_np: NumPy array of point coordinates, shape [num_points, 2 or 3].
        velocity_np: NumPy array of velocity vectors, shape [num_points, 2 or 3].

    Returns:
        NumPy array of vorticity magnitudes, shape [num_points], or zeros if calculation fails.
    """
    # Ensure PyVista is imported only when function is called, to keep it an optional dependency.
    try:
        import pyvista as pv
    except ImportError:
        print("Warning: PyVista is not installed. Cannot calculate vorticity. Returning zeros.")
        return np.zeros(points_np.shape[0], dtype=np.float32)

    try:
        if points_np.shape[0] == 0:  # No points, no vorticity
            return np.array([], dtype=np.float32)

        print(f"DEBUG_VORT_INPUT_VEL: Shape={velocity_np.shape}, AbsMean={np.abs(velocity_np).mean():.4e}, Min={velocity_np.min():.4e}, Max={velocity_np.max():.4e}, Std={velocity_np.std():.4e}")
        pv_grid = _create_pyvista_grid(points_np, velocity_np)

        if "velocity" not in pv_grid.point_data:
            print("Warning: 'velocity' field not found in PyVista grid for vorticity calculation. Returning zeros.")
            return np.zeros(points_np.shape[0], dtype=np.float32)

        # Compute derivatives including vorticity
        # The .compute_derivative() method computes vorticity and other quantities.
        # It works on PolyData (point clouds) as well.
        # We request vorticity by ensuring 'velocity' is the active vector field.
        print(f"DEBUG_VORT: Input points_np shape: {points_np.shape}, velocity_np shape: {velocity_np.shape}")
        pv_grid.active_vectors_name = 'velocity' # Ensure active vectors are explicitly set before the call
        derivative_dataset = pv_grid.compute_derivative(progress_bar=False)

        if 'gradient' in derivative_dataset.point_data:
            # PyVista stores gradient as a 9-component vector (tensor flattened row-major)
            # grad = [du/dx, du/dy, du/dz,  dv/dx, dv/dy, dv/dz,  dw/dx, dw/dy, dw/dz]
            # Indices:  0,     1,     2,      3,     4,     5,      6,     7,     8
            grad_tensor_flat = derivative_dataset.point_data['gradient']
            print(f"DEBUG_VORT: Found 'gradient' array with shape: {grad_tensor_flat.shape}")
            print(f"DEBUG_VORT: Gradient stats: abs_mean={np.abs(grad_tensor_flat).mean():.4e}, min={grad_tensor_flat.min():.4e}, max={grad_tensor_flat.max():.4e}")

            if grad_tensor_flat.shape[1] != 9:
                print(f"Warning: Gradient tensor has unexpected shape {grad_tensor_flat.shape}. Expected [N, 9]. Returning zeros for vorticity.")
                return np.zeros(points_np.shape[0], dtype=np.float32)

            # omega_x = dw/dy - dv/dz (grad[7] - grad[5])
            omega_x = grad_tensor_flat[:, 7] - grad_tensor_flat[:, 5]
            # omega_y = du/dz - dw/dx (grad[2] - grad[6])
            omega_y = grad_tensor_flat[:, 2] - grad_tensor_flat[:, 6]
            # omega_z = dv/dx - du/dy (grad[3] - grad[1])
            omega_z = grad_tensor_flat[:, 3] - grad_tensor_flat[:, 1]

            vorticity_vectors = np.stack([omega_x, omega_y, omega_z], axis=-1)
            vorticity_magnitude = np.linalg.norm(vorticity_vectors, axis=1)
            return vorticity_magnitude.astype(np.float32)
        else:
            print("Warning: 'gradient' field not found after PyVista derivative computation. Cannot calculate vorticity.")
            if derivative_dataset is not None and hasattr(derivative_dataset, 'point_data'):
                print(f"DEBUG: Available arrays in derivative_dataset point_data: {list(derivative_dataset.point_data.keys())}")
            else:
                print("DEBUG: derivative_dataset or derivative_dataset.point_data is None.")
            return np.zeros(points_np.shape[0], dtype=np.float32)

    except Exception as e:
        print(f"Error during manual vorticity calculation from gradient: {e}. Returning zeros.")
        return np.zeros(points_np.shape[0], dtype=np.float32)


if __name__ == '__main__':
    print("Testing metrics.py...")

    # Test TKE
    print("\nTesting Turbulent Kinetic Energy (TKE)...")
    vel_np = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                      dtype=np.float32)  # Magnitudes sqrt(1), sqrt(1), sqrt(1), sqrt(3)
    # Squared mags: 1, 1, 1, 3. Mean = 1.5. TKE = 0.5 * 1.5 = 0.75
    tke_np = turbulent_kinetic_energy(vel_np)
    print(f"TKE (NumPy): {tke_np}")
    assert np.isclose(tke_np, 0.75), f"TKE NumPy test failed. Expected 0.75, got {tke_np}"

    vel_torch = torch.tensor(vel_np)
    tke_torch = turbulent_kinetic_energy(vel_torch)
    print(f"TKE (Torch): {tke_torch}")
    assert np.isclose(tke_torch, 0.75), f"TKE Torch test failed. Expected 0.75, got {tke_torch}"
    print("TKE tests passed.")

    # Test Cosine Similarity
    print("\nTesting Cosine Similarity...")
    pred_cs = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.float32)  # norm sqrt(1), sqrt(2)
    targ_cs = np.array([[1, 0, 0], [0, -1, -1]], dtype=np.float32)  # norm sqrt(1), sqrt(2)
    # Sim1: (1*1)/(1*1) = 1
    # Sim2: (1*(-1) + 1*(-1)) / (sqrt(2)*sqrt(2)) = -2 / 2 = -1
    # Mean = (1 + (-1))/2 = 0

    cs_np_mean = cosine_similarity_metric(pred_cs, targ_cs, reduction='mean')
    print(f"Cosine Similarity (NumPy, mean): {cs_np_mean}")
    assert np.isclose(cs_np_mean, 0.0), "Cosine Similarity NumPy mean test failed."

    cs_np_none = cosine_similarity_metric(pred_cs, targ_cs, reduction='none')
    print(f"Cosine Similarity (NumPy, none): {cs_np_none}")
    assert np.allclose(cs_np_none, [1.0, -1.0]), "Cosine Similarity NumPy none test failed."

    pred_cs_torch = torch.tensor(pred_cs)
    targ_cs_torch = torch.tensor(targ_cs)
    cs_torch_mean = cosine_similarity_metric(pred_cs_torch, targ_cs_torch, reduction='mean')
    print(f"Cosine Similarity (Torch, mean): {cs_torch_mean}")
    assert np.isclose(cs_torch_mean, 0.0), "Cosine Similarity Torch mean test failed."

    cs_torch_none = cosine_similarity_metric(pred_cs_torch, targ_cs_torch, reduction='none')
    print(f"Cosine Similarity (Torch, none): {cs_torch_none.numpy()}")
    assert np.allclose(cs_torch_none.numpy(), [1.0, -1.0]), "Cosine Similarity Torch none test failed."
    print("Cosine Similarity tests passed.")

    # Test JSD
    print("\nTesting Jensen-Shannon Divergence (JSD)...")
    # Example from a blog: P=[0.1,0.2,0.3,0.4], Q=[0.4,0.3,0.2,0.1] -> JSD ~0.1887
    # Our function expects time series data.
    # Point 1: P1 vs Q1; Point 2: P2 vs Q2
    # For simplicity, let's make a time series where each point has a fixed distribution over time.
    # Time series for point 0: always [0.1, 0.2, 0.3, 0.4] (conceptually, after histogramming)
    # Time series for point 1: always [0.4, 0.3, 0.2, 0.1]

    # To test compute_jsd_histograms, we need magnitude series first.
    # Let's create simple magnitude series that would result in distinct histograms.
    # Point 0: real magnitudes mostly small, pred magnitudes mostly large
    # Point 1: real magnitudes mixed, pred magnitudes mixed but different
    num_t = 50
    num_p = 2
    mags_real = np.zeros((num_t, num_p))
    mags_pred = np.zeros((num_t, num_p))

    # Point 0: Real low, Pred high
    mags_real[:, 0] = np.random.uniform(0.0, 0.3, num_t)
    mags_pred[:, 0] = np.random.uniform(0.7, 1.0, num_t)
    # Point 1: Real uniform, Pred bimodal
    mags_real[:, 1] = np.random.uniform(0.0, 1.0, num_t)
    mags_pred[:num_t // 2, 1] = np.random.uniform(0.0, 0.2, num_t // 2)
    mags_pred[num_t // 2:, 1] = np.random.uniform(0.8, 1.0, num_t - num_t // 2)

    n_bins = 10
    jsd_values, pdfs_r, pdfs_p = compute_jsd_histograms(mags_real, mags_pred, num_bins=n_bins)

    print(f"JSD values per point: {jsd_values}")
    assert jsd_values.shape == (num_p,), "JSD output shape incorrect."
    assert pdfs_r.shape == (num_p, n_bins), "Real PDFs shape incorrect."
    assert pdfs_p.shape == (num_p, n_bins), "Pred PDFs shape incorrect."

    # JSD should be between 0 and log(2) approx 0.693
    assert np.all(jsd_values >= 0) and np.all(
        jsd_values <= np.log(2) + EPS), "JSD values out of expected range [0, log(2)]."
    # For Point 0, distributions should be very different, so JSD should be high.
    # For Point 1, also likely different.
    print(f"JSD for point 0 (low vs high): {jsd_values[0]:.4f}")
    print(f"JSD for point 1 (uniform vs bimodal): {jsd_values[1]:.4f}")

    # Test case where distributions are identical (JSD should be 0)
    mags_identical1 = np.random.uniform(0, 1, (num_t, num_p))
    mags_identical2 = mags_identical1.copy()
    jsd_identical, _, _ = compute_jsd_histograms(mags_identical1, mags_identical2, num_bins=n_bins)
    print(f"JSD for identical distributions: {jsd_identical}")
    assert np.allclose(jsd_identical, 0.0,
                       atol=1e-7), "JSD for identical distributions should be close to 0."  # atol due to EPS
    print("JSD tests passed.")

    # Test Vorticity (basic check for no errors and plausible output shape)
    print("\nTesting Vorticity Calculation...")
    try:
        import pyvista as pv

        # Simple 2D shear flow: u = y, v = 0. Vorticity_z = -1.
        # Points for a 2x2 square in xy plane
        points_2d = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=np.float32)
        velocity_2d = np.zeros_like(points_2d)
        velocity_2d[:, 0] = points_2d[:, 1]  # u = y

        vort_mag_2d = calculate_vorticity_magnitude(points_2d, velocity_2d)
        print(f"Vorticity magnitude (2D input): {vort_mag_2d}")
        assert vort_mag_2d is not None, "Vorticity calculation returned None for 2D input"
        assert vort_mag_2d.shape == (points_2d.shape[0],), "Vorticity (2D) output shape incorrect."
        # For u=y, v=0, w=0, curl is (0,0, d(v)/dx - d(u)/dy) = (0,0, 0 - 1) = (0,0,-1). Mag = 1.
        # PyVista's point cloud derivatives might not be perfectly accurate.
        # We're mostly checking that it runs and gives plausible values (non-negative).
        assert np.all(vort_mag_2d >= 0), "Vorticity magnitudes should be non-negative."
        print("Vorticity (2D) test ran.")

        # 3D
        points_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
        velocity_3d = np.random.rand(5, 3).astype(np.float32)
        vort_mag_3d = calculate_vorticity_magnitude(points_3d, velocity_3d)
        print(f"Vorticity magnitude (3D input): {vort_mag_3d}")
        assert vort_mag_3d is not None, "Vorticity calculation returned None for 3D input"
        assert vort_mag_3d.shape == (points_3d.shape[0],), "Vorticity (3D) output shape incorrect."
        assert np.all(vort_mag_3d >= 0), "Vorticity magnitudes should be non-negative."
        print("Vorticity (3D) test ran.")

    except ImportError:
        print("PyVista not installed, skipping vorticity calculation tests.")
    except Exception as e:
        print(f"Error during vorticity self-test: {e}")
        # If test fails, it doesn't mean the function is wrong, but test setup might be.
        # For CI, one might want to raise an error here.
    print("Vorticity tests complete (or skipped).")

    print("\nmetrics.py tests complete.")

    # Test NRMSE
    print("\nTesting NRMSE...")
    pred_nrmse_np = np.array([[1.1, 0.1, -0.1], [0.2, 1.2, 0.8]], dtype=np.float32)
    targ_nrmse_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    # Errors: [[0.1, 0.1, -0.1], [0.2, 0.2, -0.2]]
    # Squared errors per vector: [0.01+0.01+0.01=0.03], [0.04+0.04+0.04=0.12]
    # MSE = (0.03 + 0.12)/2 = 0.15/2 = 0.075. RMSE = sqrt(0.075) approx 0.27386
    # Max target norm: ||[0,1,1]|| = sqrt(2) approx 1.414
    # NRMSE_vel = 0.27386 / 1.414 approx 0.1936
    nrmse_vel_np = calculate_nrmse(pred_nrmse_np, targ_nrmse_np)
    print(f"NRMSE Velocity (NumPy): {nrmse_vel_np:.4f}")
    assert np.isclose(nrmse_vel_np, 0.1936, atol=1e-4), "NRMSE Velocity (NumPy) test failed."

    pred_nrmse_torch = torch.tensor(pred_nrmse_np)
    targ_nrmse_torch = torch.tensor(targ_nrmse_np)
    nrmse_vel_torch = calculate_nrmse(pred_nrmse_torch, targ_nrmse_torch)
    print(f"NRMSE Velocity (Torch): {nrmse_vel_torch:.4f}")
    assert np.isclose(nrmse_vel_torch, 0.1936, atol=1e-4), "NRMSE Velocity (Torch) test failed."

    # Test NRMSE for scalar field (e.g. pressure)
    pred_p_np = np.array([101.0, 102.5, 99.0], dtype=np.float32)
    targ_p_np = np.array([100.0, 102.0, 100.0], dtype=np.float32)
    # Errors: [1.0, 0.5, -1.0]
    # Squared errors: [1.0, 0.25, 1.0]. MSE = (1+0.25+1)/3 = 2.25/3 = 0.75. RMSE = sqrt(0.75) approx 0.866
    # Targ_centered: targets - mean(100.666) = [-0.666, 1.333, -0.666]
    # Max_abs_targ_centered = 1.333
    # NRMSE_p = 0.866 / 1.333 approx 0.6497
    nrmse_p_np = calculate_nrmse(pred_p_np, targ_p_np, zero_center_targets_for_pressure=True)
    print(f"NRMSE Pressure (NumPy, zero-centered): {nrmse_p_np:.4f}")
    assert np.isclose(nrmse_p_np, 0.6497, atol=1e-4), "NRMSE Pressure (NumPy) test failed."

    pred_p_torch = torch.tensor(pred_p_np)
    targ_p_torch = torch.tensor(targ_p_np)
    nrmse_p_torch = calculate_nrmse(pred_p_torch, targ_p_torch, zero_center_targets_for_pressure=True)
    print(f"NRMSE Pressure (Torch, zero-centered): {nrmse_p_torch:.4f}")
    assert np.isclose(nrmse_p_torch, 0.6497, atol=1e-4), "NRMSE Pressure (Torch) test failed."
    print("NRMSE tests passed.")


def calculate_nrmse(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    zero_center_targets_for_pressure: bool = False,
    epsilon: float = EPS
) -> float:
    """
    Calculates the Normalized Root Mean Square Error (NRMSE).
    NRMSE = RMSE / (max(targets) - min(targets)) or RMSE / max|targets| depending on context.
    Using max|targets| as per the paper: (1 / max|u_gt|) * sqrt( (1/N) * Σ ||u_pred - u_gt||² )
    For pressure, targets are zero-centered first if specified.

    Args:
        predictions: Predicted values, shape [num_points, num_components] or [num_points].
        targets: Target values, same shape as predictions.
        zero_center_targets_for_pressure: If True, subtracts mean from targets before calculating max abs value
                                         (used for pressure NRMSE as per paper).
        epsilon: Small value to prevent division by zero in normalization.

    Returns:
        Scalar NRMSE value.
    """
    is_torch = isinstance(predictions, torch.Tensor)

    if is_torch:
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets).to(predictions.device).type_as(predictions)

        if zero_center_targets_for_pressure:
            # This is for pressure NRMSE. Targets are zero-centered at each time step (not handled here).
            # The paper says: "pressure NRMSE then assumes the same form as velocity NRMSE"
            # "we shifted the predicted pressure values by the difference in predicted and ground truth means."
            # "pressure NRMSE values are mean-normalised" for 3D case.
            # Let's assume for this function, if zero_center is true, we do it globally on the input targets.
            targets_for_norm = targets - torch.mean(targets)
        else:
            targets_for_norm = targets

        # RMSE part: sqrt(mean_squared_error)
        # If predictions/targets are vectors [N, D], mse is mean of sum of squared errors per vector.
        if predictions.ndim > 1 and predictions.shape[-1] > 1: # Vector field
            squared_error = (predictions - targets).pow(2).sum(dim=-1) # Sum over components
            mse = torch.mean(squared_error) # Mean over N points
        else: # Scalar field
            mse = torch.mean((predictions - targets).pow(2))
        rmse = torch.sqrt(mse)

        # Normalization factor: max absolute value of targets (or zero-centered targets)
        # For vector fields like velocity, max|u_gt| means max of vector norms.
        if targets_for_norm.ndim > 1 and targets_for_norm.shape[-1] > 1: # Vector field
            max_abs_target = torch.linalg.norm(targets_for_norm, dim=-1).max()
        else: # Scalar field
            max_abs_target = torch.abs(targets_for_norm).max()

        if max_abs_target < epsilon: # Avoid division by zero if targets are all zero
            return float(rmse.item()) # Return raw RMSE if normalization factor is zero

        nrmse = rmse / max_abs_target
        return float(nrmse.item())

    else: # Numpy
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets, dtype=predictions.dtype)

        if zero_center_targets_for_pressure:
            targets_for_norm = targets - np.mean(targets)
        else:
            targets_for_norm = targets

        if predictions.ndim > 1 and predictions.shape[-1] > 1: # Vector field
            squared_error = np.sum((predictions - targets)**2, axis=-1)
            mse = np.mean(squared_error)
        else: # Scalar field
            mse = np.mean((predictions - targets)**2)
        rmse = np.sqrt(mse)

        if targets_for_norm.ndim > 1 and targets_for_norm.shape[-1] > 1: # Vector field
            max_abs_target = np.max(np.linalg.norm(targets_for_norm, axis=-1))
        else: # Scalar field
            max_abs_target = np.max(np.abs(targets_for_norm))

        if max_abs_target < epsilon:
            return float(rmse)

        nrmse = rmse / max_abs_target
        return float(nrmse)


def calculate_perc_points_within_rel_error(
    true_vel_np: np.ndarray,
    pred_vel_np: np.ndarray,
    rel_tolerance: float = 0.1,
    epsilon: float = 1e-9 # To avoid division by zero if true_vel component is zero
) -> tuple[float, np.ndarray]:
    """
    Calculates the percentage of points where all velocity components (x, y, z)
    are within a specified relative error tolerance.
    Relative error = |true - pred| / (|true| + epsilon)

    Args:
        true_vel_np: NumPy array of true velocity vectors, shape [num_points, 3].
        pred_vel_np: NumPy array of predicted velocity vectors, shape [num_points, 3].
        rel_tolerance: The relative error tolerance (e.g., 0.1 for 10%).
        epsilon: Small value to prevent division by zero.

    Returns:
        A tuple containing:
            - percentage_compliant_points: Float, percentage of points meeting the criteria (0.0 to 100.0).
            - point_wise_compliance: NumPy boolean array of shape [num_points],
                                     True if the point meets the criteria, False otherwise.
    """
    if true_vel_np.shape != pred_vel_np.shape:
        raise ValueError("True and predicted velocity arrays must have the same shape.")
    if true_vel_np.ndim != 2 or true_vel_np.shape[1] != 3:
        # Allow for 2D velocity vectors as well, by checking if last dim is 2 or 3
        if not (true_vel_np.ndim == 2 and true_vel_np.shape[1] in [2,3]):
             raise ValueError(f"Velocity arrays must be of shape [num_points, 2 or 3], got {true_vel_np.shape}")

    num_points = true_vel_np.shape[0]
    if num_points == 0:
        return 0.0, np.array([], dtype=bool)

    # Calculate relative error for each component
    # rel_error_vx = |true_vx - pred_vx| / (|true_vx| + eps)
    # rel_error_vy = |true_vy - pred_vy| / (|true_vy| + eps)
    # rel_error_vz = |true_vz - pred_vz| / (|true_vz| + eps)
    abs_true_vel = np.abs(true_vel_np)
    abs_error = np.abs(true_vel_np - pred_vel_np)

    relative_errors = abs_error / (abs_true_vel + epsilon)

    # Check compliance for each component
    component_wise_compliance = relative_errors <= rel_tolerance # Shape [num_points, num_components]

    # A point is compliant if ALL its components are compliant
    point_wise_compliance = np.all(component_wise_compliance, axis=1) # Shape [num_points]

    num_compliant_points = np.sum(point_wise_compliance)
    percentage_compliant_points = (num_compliant_points / num_points) * 100.0 if num_points > 0 else 0.0

    return float(percentage_compliant_points), point_wise_compliance


# --- Slice Analysis ---

def get_slice_indices(
    points: np.ndarray, # Shape [N, 3]
    axis_idx: int,      # 0 for X, 1 for Y, 2 for Z
    position: float,    # Coordinate value for the slice center
    thickness: float    # Thickness of the slice
) -> np.ndarray:
    """
    Returns indices of points that fall within a slice of given thickness centered at position along an axis.

    Args:
        points: NumPy array of point coordinates, shape [N, 3].
        axis_idx: Index of the axis to slice along (0 for X, 1 for Y, 2 for Z).
        position: Coordinate value for the center of the slice.
        thickness: Total thickness of the slice.

    Returns:
        NumPy array of indices of points within the slice.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be a [N, 3] array.")
    if not 0 <= axis_idx <= 2:
        raise ValueError("axis_idx must be 0, 1, or 2.")
    if thickness < 0:
        raise ValueError("thickness must be non-negative.")

    half_thickness = thickness / 2.0
    lower_bound = position - half_thickness
    upper_bound = position + half_thickness

    point_coords_on_axis = points[:, axis_idx]

    slice_indices = np.where(
        (point_coords_on_axis >= lower_bound) & (point_coords_on_axis <= upper_bound)
    )[0]

    return slice_indices.astype(int)


def calculate_slice_metrics_for_frame(
    points_np: np.ndarray,           # Shape [N, 3]
    true_velocity_np: np.ndarray,    # Shape [N, 3]
    pred_velocity_np: np.ndarray,    # Shape [N, 3]
    slice_config: dict               # From global config: e.g. {"axes": ["X", "Y"], "num_per_axis": 3, "thickness_percent": 1.0}
) -> dict:
    """
    Calculates velocity metrics (max_mag, avg_mag) for specified slices of a single frame.

    Args:
        points_np: NumPy array of point coordinates.
        true_velocity_np: NumPy array of true velocities.
        pred_velocity_np: NumPy array of predicted velocities.
        slice_config: Dictionary containing slice analysis parameters.

    Returns:
        A dictionary containing metrics per slice.
        Example: {"slice_X_0_pos_0.12_max_true_vel": 0.5, "slice_X_0_pos_0.12_max_pred_vel": 0.45, ...}
    """
    if points_np.shape[0] == 0:
        return {} # No points, no slices

    results = {}
    axis_map = {"X": 0, "Y": 1, "Z": 2}

    # Calculate bounding box for determining slice positions and thickness
    min_coords = points_np.min(axis=0)
    max_coords = points_np.max(axis=0)
    bbox_extents = max_coords - min_coords

    for axis_name in slice_config.get("axes", ["X", "Y", "Z"]):
        axis_idx = axis_map.get(axis_name)
        if axis_idx is None:
            print(f"Warning: Invalid axis '{axis_name}' in slice_config. Skipping.")
            continue

        if bbox_extents[axis_idx] < EPS: # Axis has no extent (e.g. 2D data for Z slice)
            # print(f"Warning: Bounding box extent for axis {axis_name} is near zero. Skipping slices for this axis.")
            continue

        num_slices = slice_config.get("num_per_axis", 3)
        thickness_abs = (slice_config.get("thickness_percent", 1.0) / 100.0) * bbox_extents[axis_idx]
        if thickness_abs < EPS: # If thickness is effectively zero, make it a small value relative to extent
            thickness_abs = 0.001 * bbox_extents[axis_idx] if bbox_extents[axis_idx] > EPS else 0.001


        # Determine slice positions
        if num_slices == 1:
            positions = [min_coords[axis_idx] + 0.5 * bbox_extents[axis_idx]]
        else:
            # Positions from (1 / (num_slices+1)) to (num_slices / (num_slices+1)) of extent
            # e.g., for 3 slices: 0.25, 0.50, 0.75
            positions = [min_coords[axis_idx] + (i + 1) * (bbox_extents[axis_idx] / (num_slices + 1)) for i in range(num_slices)]

        for i, slice_pos in enumerate(positions):
            slice_indices = get_slice_indices(points_np, axis_idx, slice_pos, thickness_abs)

            slice_key_prefix = f"slice_{axis_name}_{i}_pos{slice_pos:.2f}"

            if slice_indices.size > 0:
                true_vel_slice = true_velocity_np[slice_indices]
                pred_vel_slice = pred_velocity_np[slice_indices]

                true_vel_mag_slice = np.linalg.norm(true_vel_slice, axis=1)
                pred_vel_mag_slice = np.linalg.norm(pred_vel_slice, axis=1)

                results[f"{slice_key_prefix}_max_true_vel_mag"] = float(true_vel_mag_slice.max())
                results[f"{slice_key_prefix}_avg_true_vel_mag"] = float(true_vel_mag_slice.mean())
                results[f"{slice_key_prefix}_max_pred_vel_mag"] = float(pred_vel_mag_slice.max())
                results[f"{slice_key_prefix}_avg_pred_vel_mag"] = float(pred_vel_mag_slice.mean())
                results[f"{slice_key_prefix}_num_points"] = len(slice_indices)
            else:
                # Log NaN or skip if no points in slice? For now, log NaN for consistency if key expected.
                results[f"{slice_key_prefix}_max_true_vel_mag"] = np.nan
                results[f"{slice_key_prefix}_avg_true_vel_mag"] = np.nan
                results[f"{slice_key_prefix}_max_pred_vel_mag"] = np.nan
                results[f"{slice_key_prefix}_avg_pred_vel_mag"] = np.nan
                results[f"{slice_key_prefix}_num_points"] = 0
    return results


def calculate_velocity_gradients(points_np: np.ndarray, velocity_np: np.ndarray) -> np.ndarray | None:
    """
    Calculates the gradient tensor of the velocity field at each point.
    Uses PyVista for the underlying computation.
    The gradient tensor is returned as a [num_points, 9] array (row-major order:
    du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz).

    Args:
        points_np: NumPy array of point coordinates, shape [num_points, 2 or 3].
        velocity_np: NumPy array of velocity vectors, shape [num_points, 2 or 3].

    Returns:
        NumPy array of shape [num_points, 9] containing the flattened gradient tensor
        for each point, or None if calculation fails.
        For 2D inputs (points [N,2], velocity [N,2]), the z-components of gradients
        (e.g. du/dz, dv/dz, dw/dx, dw/dy, dw/dz) will be zero. dw/dx, dw/dy will be zero
        if velocity_np is [N,2] (as w component is zero).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("Warning: PyVista is not installed. Cannot calculate velocity gradients. Returning None.")
        return None

    if points_np.shape[0] == 0:
        return np.empty((0, 9), dtype=np.float32)
    if points_np.shape[0] != velocity_np.shape[0]:
        raise ValueError("Points and velocity arrays must have the same number of points for gradient calculation.")

    try:
        # _create_pyvista_grid handles 2D/3D conversion and sets up the grid
        pv_grid = _create_pyvista_grid(points_np, velocity_np)

        if "velocity" not in pv_grid.point_data:
            print("Warning: 'velocity' field not found in PyVista grid for gradient calculation. Returning None.")
            return None

        # Compute derivatives. The 'gradient' field will be a 9-component vector (tensor flattened row-wise).
        # For 3D velocity U=(u,v,w) and points (x,y,z):
        # gradient = [du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz]
        pv_grid.active_vectors_name = 'velocity' # Ensure active vectors are explicitly set before the call
        derivative_dataset = pv_grid.compute_derivative(progress_bar=False)

        if 'gradient' in derivative_dataset.point_data:
            grad_tensor_flat = derivative_dataset.point_data['gradient'] # Shape [num_points, 9]

            # Ensure output is always [N,9], even if input was 2D points/velocity.
            # PyVista's derivative filter should handle this correctly for its 'gradient' output.
            # If input velocity was [N,2] (i.e. w=0), then dw/dx, dw/dy, dw/dz should be zero.
            # If input points were [N,2] (i.e. z=0 effectively), then du/dz, dv/dz might be zero or reflect assumption of 2D flow.
            # The exact behavior for 2D inputs needs to be noted if it's critical.
            # For now, we assume PyVista provides a consistent [N,9] output.
            if grad_tensor_flat.shape[1] != 9:
                 print(f"Warning: PyVista gradient output shape is {grad_tensor_flat.shape}, expected [N, 9]. Returning None.")
                 return None
            return grad_tensor_flat.astype(np.float32)
        else:
            print("Warning: 'gradient' field not found after PyVista derivative computation.")
            if derivative_dataset is not None and hasattr(derivative_dataset, 'point_data'):
                print(f"DEBUG: Available arrays in derivative_dataset point_data: {list(derivative_dataset.point_data.keys())}")
            else:
                print("DEBUG: derivative_dataset or derivative_dataset.point_data is None.")
            return None

    except Exception as e:
        print(f"Error during PyVista velocity gradient calculation: {e}. Returning None.")
        return None
