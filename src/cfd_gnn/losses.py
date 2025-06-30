# -*- coding: utf-8 -*-
"""
losses.py
---------
Loss functions for training CFD GNN models, including supervised loss,
physics-informed divergence loss, and histogram-based loss.
"""
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.data import Data

def calculate_divergence(predicted_velocity: torch.Tensor, graph_data: Data) -> torch.Tensor:
    """
    Calculates the divergence of the predicted velocity field on the graph.
    div(U) approx sum_j (U_i_predicted * (pos_j - pos_i)) for edges (i,j) connected to node i.
    More accurately, using flux: sum_j (U_src * edge_attr_src_to_dst).sum(-1) aggregated at dst.

    Args:
        predicted_velocity: Tensor of shape [num_nodes, 3] representing predicted velocities.
        graph_data: PyTorch Geometric Data object containing edge_index and edge_attr
                    (relative positions from source to target).

    Returns:
        A tensor of shape [num_nodes] representing the divergence at each node.
    """
    src_nodes, dst_nodes = graph_data.edge_index

    # edge_attr stores (pos[dst] - pos[src]), which is the vector along the edge
    # For flux U_src · (pos_dst - pos_src), this is correct.
    # Or, if edge_attr is (pos_src - pos_dst), then flux is -U_src · (pos_src - pos_dst)
    # The original script had `rel = pts[dst] - pts[src]`, so edge_attr is (pos[dst] - pos[src])
    # Flux from src to dst along the edge: (predicted_velocity[src_nodes] * graph_data.edge_attr).sum(dim=-1)

    # Ensure edge_attr is on the same device as predicted_velocity
    edge_attr_device = graph_data.edge_attr.to(predicted_velocity.device)

    # Calculate flux across each edge: dot product of velocity at source node and edge vector
    # Flux_ij = U_i · (x_j - x_i)
    flux_values = (predicted_velocity[src_nodes] * edge_attr_device).sum(dim=-1)

    # Aggregate (sum) fluxes for each destination node
    # This approximates integral(U·n dA) over the surface of a control volume around the node.
    # For scatter_add, index should be the target node for incoming flux.
    divergence_at_nodes = scatter_add(
        flux_values,
        dst_nodes, # group by destination node
        dim=0,
        dim_size=predicted_velocity.size(0) # ensure output size matches number of nodes
    )
    return divergence_at_nodes


def physics_informed_divergence_loss(
    predicted_velocity: torch.Tensor,
    graph_data: Data,
    target_divergence: torch.Tensor | None = None, # Optional: if non-zero target div
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Calculates a loss based on the divergence of the predicted velocity field.
    Typically, for incompressible flow, divergence should be close to zero.
    Loss = mean(divergence_at_nodes^2) or specified reduction.

    Args:
        predicted_velocity: Tensor of shape [num_nodes, 3].
        graph_data: PyTorch Geometric Data object.
        target_divergence: Optional target divergence field (e.g., if there are sources/sinks).
                           Defaults to zeros.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss tensor.
    """
    divergence = calculate_divergence(predicted_velocity, graph_data)

    if target_divergence is None:
        target_divergence = torch.zeros_like(divergence)

    # Ensure target_divergence is on the same device
    target_divergence = target_divergence.to(divergence.device)

    # Loss is typically the squared deviation from target (usually zero)
    loss = (divergence - target_divergence).pow(2)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")


def wasserstein1_histogram_loss(
    divergence_values: torch.Tensor,
    num_bins: int,
    epsilon: float = 1e-9 # For numerical stability in CDF
) -> torch.Tensor:
    """
    Calculates a Wasserstein-1 like distance for the histogram of divergence values.
    This encourages the distribution of divergence values to be centered around zero.
    The loss is the mean absolute difference between the empirical CDF of divergence
    and the CDF of a uniform distribution (implicitly, by comparing to a straight line
    from 0 to 1 after sorting, or by comparing cumulative histogram to target).

    The original implementation in `train_and_validate_with_noise.py` computes:
    `torch.mean(torch.abs(torch.cumsum(h, 0) - torch.linspace(0, 1, nbins)))`
    where `h` is the density histogram. This measures deviation of empirical CDF from
    a uniform distribution's CDF over the histogram range.

    Args:
        divergence_values: Tensor of shape [num_nodes] representing divergence at each node.
                           Should be detached from the computation graph for histogramming.
        num_bins: Number of bins for the histogram.
        epsilon: Small value for numerical stability.

    Returns:
        Scalar loss tensor.
    """
    if divergence_values.numel() == 0: # Handle empty tensor case
        return torch.tensor(0.0, device=divergence_values.device, dtype=torch.float32)

    # Detach divergence values as histogram creation is not differentiable directly by PyTorch
    d_detached = divergence_values.detach().cpu() # Histogramming is often easier on CPU

    # Determine range for histogram: symmetric around zero based on max absolute value
    # Add epsilon to tau to ensure bins are not degenerate if all values are zero
    tau = float(d_detached.abs().max()) + epsilon
    if tau <= epsilon : # if all values were zero or very close
        tau = epsilon + 1.0 # set a default range to avoid issues with linspace

    # Create histogram
    # density=True normalizes histogram so that the sum of bar areas equals 1
    hist_counts, _ = torch.histogram(
        d_detached,
        bins=num_bins,
        range=(-tau, tau), # Symmetric range around 0
        density=True
    ) # hist_counts is shape [num_bins]

    # Calculate empirical CDF from the density histogram
    # The width of each bin is (2*tau) / num_bins
    # Cumulative sum of (count * bin_width) gives the CDF values at bin edges
    bin_width = (2 * tau) / num_bins
    empirical_cdf = torch.cumsum(hist_counts * bin_width, dim=0)

    # Target CDF for a distribution perfectly centered at zero and symmetric
    # would be a step function. The original code compared to `torch.linspace(0, 1, nbins)`,
    # which is the CDF of a uniform distribution over the *binned range*.
    # This encourages the divergence values to be spread out if using that target.
    # A more direct target for "centered around zero" might involve a different comparison.
    # Replicating the original paper's/code's intent:
    target_cdf_points = torch.linspace(0.0, 1.0, num_bins, device=empirical_cdf.device)

    # The comparison `torch.abs(empirical_cdf - target_cdf_points)` measures the difference
    # at the *end* of each bin.
    # The original code `torch.cumsum(h,0)` where `h` is density gives the CDF values.
    # Then `torch.mean(torch.abs(cdf - linspace(0,1,nbins)))`. This is L1 distance between CDFs.

    loss = torch.mean(torch.abs(empirical_cdf - target_cdf_points))

    return loss.to(divergence_values.device) # Move loss to original device


def combined_loss(
    predicted_velocity: torch.Tensor,
    true_velocity: torch.Tensor,
    graph_data: Data,
    loss_weights: dict, # e.g., {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
    histogram_bins: int = 64,
    divergence_target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Calculates a combined loss for training the GNN.

    Args:
        predicted_velocity: Predicted velocity tensor [num_nodes, 3].
        true_velocity: Ground truth velocity tensor [num_nodes, 3].
        graph_data: PyTorch Geometric Data object.
        loss_weights: Dictionary containing weights for different loss components.
        histogram_bins: Number of bins for the histogram loss.
        divergence_target: Optional target for divergence (defaults to zero).

    Returns:
        A tuple containing:
            - total_loss: The combined scalar loss.
            - individual_losses: A dictionary of the individual loss components.
    """
    individual_losses = {}

    # 1. Supervised Loss (MSE on velocity)
    # Ensure true_velocity is on the same device as predicted_velocity
    true_velocity_device = true_velocity.to(predicted_velocity.device)
    loss_supervised = F.mse_loss(predicted_velocity, true_velocity_device)
    individual_losses["supervised"] = loss_supervised

    # 2. Physics-Informed Divergence Loss
    # Calculate divergence from the *predicted* velocity
    divergence_values_pred = calculate_divergence(predicted_velocity, graph_data)
    loss_divergence = (divergence_values_pred - (divergence_target or 0.0)).pow(2).mean()
    individual_losses["divergence"] = loss_divergence

    # 3. Histogram Loss (on the divergence of the prediction)
    loss_histogram = wasserstein1_histogram_loss(divergence_values_pred, histogram_bins)
    individual_losses["histogram"] = loss_histogram

    # Combine losses using weights
    total_loss = torch.tensor(0.0, device=predicted_velocity.device)
    if "supervised" in loss_weights and loss_weights["supervised"] > 0:
        total_loss += loss_weights["supervised"] * loss_supervised
    if "divergence" in loss_weights and loss_weights["divergence"] > 0:
        total_loss += loss_weights["divergence"] * loss_divergence
    if "histogram" in loss_weights and loss_weights["histogram"] > 0:
        total_loss += loss_weights["histogram"] * loss_histogram

    # Store weighted losses for logging if needed
    individual_losses["weighted_supervised"] = loss_weights.get("supervised", 0.0) * loss_supervised
    individual_losses["weighted_divergence"] = loss_weights.get("divergence", 0.0) * loss_divergence
    individual_losses["weighted_histogram"] = loss_weights.get("histogram", 0.0) * loss_histogram

    return total_loss, individual_losses


if __name__ == '__main__':
    print("Testing losses.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy data for testing
    num_nodes = 20
    num_edges = 50
    pred_vel = torch.randn(num_nodes, 3, device=device, requires_grad=True)
    true_vel = torch.randn(num_nodes, 3, device=device)

    # Create realistic-looking edge_index and edge_attr
    edge_idx = torch.randint(0, num_nodes, (2, num_edges), device=device)
    # Ensure no self-loops for simplicity in divergence test, though scatter_add handles it
    edge_idx = edge_idx[:, edge_idx[0] != edge_idx[1]]
    num_edges = edge_idx.shape[1]

    pos = torch.randn(num_nodes, 3, device=device)
    edge_attr_test = pos[edge_idx[1]] - pos[edge_idx[0]] # Target - Source

    test_graph_data = Data(
        x=true_vel, # Not used by divergence directly, but part of graph
        pos=pos,
        edge_index=edge_idx,
        edge_attr=edge_attr_test
    ).to(device)

    # Test calculate_divergence
    print("\nTesting calculate_divergence...")
    div_values = calculate_divergence(pred_vel, test_graph_data)
    print(f"Divergence values shape: {div_values.shape} (Expected: [{num_nodes}])")
    assert div_values.shape == (num_nodes,)
    print(f"Sample divergence values: {div_values[:5].tolist()}")

    # Test physics_informed_divergence_loss
    print("\nTesting physics_informed_divergence_loss...")
    phy_loss = physics_informed_divergence_loss(pred_vel, test_graph_data)
    print(f"Physics divergence loss: {phy_loss.item()}")
    assert phy_loss.ndim == 0

    # Test wasserstein1_histogram_loss
    print("\nTesting wasserstein1_histogram_loss...")
    # Use a clone of div_values as it should be detached
    hist_loss = wasserstein1_histogram_loss(div_values.clone(), num_bins=32)
    print(f"Histogram loss: {hist_loss.item()}")
    assert hist_loss.ndim == 0

    # Test with all zero divergence
    zero_div_values = torch.zeros(num_nodes, device=device)
    hist_loss_zero_div = wasserstein1_histogram_loss(zero_div_values, num_bins=32)
    print(f"Histogram loss (zero divergence): {hist_loss_zero_div.item()}")
    # For perfectly zero divergence, the histogram should be a spike at 0.
    # The W1 distance to a uniform distribution over the range could be non-zero.
    # The original loss formulation might penalize perfectly zero divergence if range is small.

    # Test combined_loss
    print("\nTesting combined_loss...")
    loss_w = {"supervised": 1.0, "divergence": 0.1, "histogram": 0.05}
    total_l, individual_ls = combined_loss(pred_vel, true_vel, test_graph_data, loss_w, histogram_bins=32)
    print(f"Total combined loss: {total_l.item()}")
    print(f"Individual losses: { {k: v.item() for k, v in individual_ls.items()} }")
    assert total_l.ndim == 0
    assert "supervised" in individual_ls
    assert "divergence" in individual_ls
    assert "histogram" in individual_ls

    # Test backward pass
    try:
        total_l.backward()
        print("Backward pass successful.")
        assert pred_vel.grad is not None, "Gradients not computed for pred_vel."
        print(f"Sample gradient for pred_vel: {pred_vel.grad[0].tolist()}")
    except Exception as e:
        print(f"Error during backward pass: {e}")
        raise

    print("\nlosses.py tests passed.")
