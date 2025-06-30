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
        squared_magnitudes = torch.sum(velocity_field**2, dim=-1)
        tke_val = 0.5 * torch.mean(squared_magnitudes)
        return float(tke_val.item())
    elif isinstance(velocity_field, np.ndarray):
        squared_magnitudes = np.sum(velocity_field**2, axis=-1)
        tke_val = 0.5 * np.mean(squared_magnitudes)
        return float(tke_val)
    else:
        raise TypeError("Input velocity_field must be a NumPy array or PyTorch tensor.")


def cosine_similarity_metric(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    reduction: str = 'mean' # 'mean' or 'none'
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
    else: # Numpy
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
    min_max_range: tuple[float, float] | None = None, # Optional: precomputed (min, max) for histogram bins
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
        max_val = min_val + EPS # Add epsilon if min and max are too close or equal

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
        kl_pm = np.sum(p_dist * np.log(p_dist / m_dist)) # log base e
        kl_qm = np.sum(q_dist * np.log(q_dist / m_dist))

        jsd_per_point[i] = 0.5 * (kl_pm + kl_qm)

    return jsd_per_point, pdfs_real, pdfs_pred


if __name__ == '__main__':
    print("Testing metrics.py...")

    # Test TKE
    print("\nTesting Turbulent Kinetic Energy (TKE)...")
    vel_np = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,1]], dtype=np.float32) # Magnitudes sqrt(1), sqrt(1), sqrt(1), sqrt(3)
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
    pred_cs = np.array([[1,0,0], [0,1,1]], dtype=np.float32) # norm sqrt(1), sqrt(2)
    targ_cs = np.array([[1,0,0], [0,-1,-1]], dtype=np.float32) # norm sqrt(1), sqrt(2)
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
    mags_pred[:num_t//2, 1] = np.random.uniform(0.0, 0.2, num_t//2)
    mags_pred[num_t//2:, 1] = np.random.uniform(0.8, 1.0, num_t - num_t//2)

    n_bins = 10
    jsd_values, pdfs_r, pdfs_p = compute_jsd_histograms(mags_real, mags_pred, num_bins=n_bins)

    print(f"JSD values per point: {jsd_values}")
    assert jsd_values.shape == (num_p,), "JSD output shape incorrect."
    assert pdfs_r.shape == (num_p, n_bins), "Real PDFs shape incorrect."
    assert pdfs_p.shape == (num_p, n_bins), "Pred PDFs shape incorrect."

    # JSD should be between 0 and log(2) approx 0.693
    assert np.all(jsd_values >= 0) and np.all(jsd_values <= np.log(2) + EPS), "JSD values out of expected range [0, log(2)]."
    # For Point 0, distributions should be very different, so JSD should be high.
    # For Point 1, also likely different.
    print(f"JSD for point 0 (low vs high): {jsd_values[0]:.4f}")
    print(f"JSD for point 1 (uniform vs bimodal): {jsd_values[1]:.4f}")

    # Test case where distributions are identical (JSD should be 0)
    mags_identical1 = np.random.uniform(0, 1, (num_t, num_p))
    mags_identical2 = mags_identical1.copy()
    jsd_identical, _, _ = compute_jsd_histograms(mags_identical1, mags_identical2, num_bins=n_bins)
    print(f"JSD for identical distributions: {jsd_identical}")
    assert np.allclose(jsd_identical, 0.0, atol=1e-7), "JSD for identical distributions should be close to 0." # atol due to EPS
    print("JSD tests passed.")

    print("\nmetrics.py tests complete.")
