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
        dst_nodes,  # group by destination node
        dim=0,
        dim_size=predicted_velocity.size(0)  # ensure output size matches number of nodes
    )
    return divergence_at_nodes


def physics_informed_divergence_loss(
        predicted_velocity: torch.Tensor,
        graph_data: Data,
        target_divergence: torch.Tensor | None = None,  # Optional: if non-zero target div
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
        epsilon: float = 1e-9  # For numerical stability in CDF
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
    if divergence_values.numel() == 0:  # Handle empty tensor case
        return torch.tensor(0.0, device=divergence_values.device, dtype=torch.float32)

    # Detach divergence values as histogram creation is not differentiable directly by PyTorch
    d_detached = divergence_values.detach().cpu()  # Histogramming is often easier on CPU

    # Determine range for histogram: symmetric around zero based on max absolute value
    # Add epsilon to tau to ensure bins are not degenerate if all values are zero
    tau = float(d_detached.abs().max()) + epsilon
    if tau <= epsilon:  # if all values were zero or very close
        tau = epsilon + 1.0  # set a default range to avoid issues with linspace

    # Create histogram
    # density=True normalizes histogram so that the sum of bar areas equals 1
    hist_counts, _ = torch.histogram(
        d_detached,
        bins=num_bins,
        range=(-tau, tau),  # Symmetric range around 0
        density=True
    )  # hist_counts is shape [num_bins]

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

    return loss.to(divergence_values.device)  # Move loss to original device


def combined_loss(
        model_output_t1: torch.Tensor,  # Model output at t1, shape [N, 4] (vel_x, vel_y, vel_z, pressure)
        true_velocity_t1: torch.Tensor,  # Ground truth velocity at t1, shape [N, 3]
        graph_t0: Data,  # Graph data at t0 (for N-S: u_true_t0, time_t0)
        graph_t1: Data,  # Graph data at t1 (for N-S: time_t1, and spatial structure for derivatives)
        loss_weights: dict,  # e.g., {"supervised": 1.0, "divergence": 0.1, "navier_stokes": 0.01, "lbc": 0.5}
        reynolds_number: float | None = None,
        histogram_bins: int = 64,
        divergence_target: torch.Tensor | None = None,
        # Boundary condition related arguments (optional)
        boundary_nodes_mask: torch.Tensor | None = None,
        target_boundary_velocity: torch.Tensor | float = 0.0
) -> tuple[torch.Tensor, dict]:
    """
    Calculates a combined loss for training the GNN.

    Args:
        model_output_t1: Predicted output from the model [num_nodes, 4 (vx,vy,vz,p)].
        true_velocity_t1: Ground truth velocity tensor [num_nodes, 3].
        graph_t0: PyTorch Geometric Data object for the previous timestep (t0).
        graph_t1: PyTorch Geometric Data object for the current timestep (t1).
                  Used for spatial derivatives in N-S loss and divergence calculation.
                  Can also contain `boundary_nodes_mask` if static, or it's passed separately.
        loss_weights: Dictionary containing weights for different loss components.
        reynolds_number: Reynolds number, required if 'navier_stokes' weight > 0.
        histogram_bins: Number of bins for the histogram loss.
        divergence_target: Optional target for divergence (defaults to zero).
        boundary_nodes_mask: Optional boolean tensor [N] indicating boundary nodes.
        target_boundary_velocity: Optional target velocity for boundary nodes. Scalar or [N_boundary, 3].

    Returns:
        A tuple containing:
            - total_loss: The combined scalar loss.
            - individual_losses: A dictionary of the individual loss components.
    """
    individual_losses = {}
    predicted_velocity_t1 = model_output_t1[:, :3]  # First 3 components are velocity

    # 1. Supervised Loss (MSE on velocity)
    true_velocity_t1_device = true_velocity_t1.to(predicted_velocity_t1.device)
    loss_supervised = F.mse_loss(predicted_velocity_t1, true_velocity_t1_device)
    individual_losses["supervised"] = loss_supervised

    # Graph structure for divergence and N-S spatial derivatives is from graph_t1
    # (or graph_t0 if structure is static and passed as graph_t1 for this purpose)
    current_graph_structure = graph_t1

    # 2. Physics-Informed Divergence Loss (Continuity part of N-S for u_pred_t1)
    # This is calculated on the predicted velocity at t1.
    divergence_values_pred_t1 = calculate_divergence(predicted_velocity_t1, current_graph_structure)
    loss_divergence = (divergence_values_pred_t1 - (divergence_target or 0.0)).pow(2).mean()
    individual_losses["divergence"] = loss_divergence  # This is effectively the continuity loss from N-S
    individual_losses["divergence_values_pred_for_debug"] = divergence_values_pred_t1

    # 3. Histogram Loss (on the divergence of the prediction at t1)
    loss_histogram = wasserstein1_histogram_loss(divergence_values_pred_t1, histogram_bins)
    individual_losses["histogram"] = loss_histogram

    # 4. Navier-Stokes Momentum Loss
    # The navier_stokes_loss function internally calculates continuity again, but we can use its momentum part.
    # Or, we can have navier_stokes_loss return momentum and continuity parts separately.
    # For now, let's assume navier_stokes_loss returns (total_ns, momentum_res_loss, continuity_res_loss)
    # We will use its momentum part and our already computed divergence for continuity.

    use_ns_loss = "navier_stokes" in loss_weights and loss_weights["navier_stokes"] > 0
    if use_ns_loss:
        if reynolds_number is None:
            raise ValueError("Reynolds number must be provided if Navier-Stokes loss is active.")
        # navier_stokes_loss takes the full model_output_t1 (which includes pressure)
        _, loss_ns_momentum, _ = navier_stokes_loss(
            model_output_t1, graph_t0, graph_t1, reynolds_number, reduction='mean'
        )
        # The continuity part is already handled by loss_divergence if we consider div(u_pred_t1).
        # The N-S paper's LPDE = ||F_mom||^2 + ||F_cont||^2. So, we add momentum here.
        # The existing "divergence" loss IS the continuity loss.
        individual_losses["navier_stokes_momentum"] = loss_ns_momentum
    else:
        individual_losses["navier_stokes_momentum"] = torch.tensor(0.0, device=model_output_t1.device)

    # 5. Boundary Condition Loss (LBC)
    use_lbc_loss = "lbc" in loss_weights and loss_weights["lbc"] > 0
    if use_lbc_loss:
        # If boundary_nodes_mask is stored in graph_t1 (e.g. if static for the geometry)
        # This is just an example, the mask could also be passed directly if it varies or comes from elsewhere.
        actual_boundary_mask = boundary_nodes_mask
        if actual_boundary_mask is None and hasattr(graph_t1, 'boundary_mask'):
            actual_boundary_mask = graph_t1.boundary_mask

        loss_lbc = boundary_condition_loss(
            predicted_velocity_t1,
            actual_boundary_mask,  # Use the resolved mask
            target_boundary_velocity,  # Passed directly
            reduction='mean'
        )
        individual_losses["lbc"] = loss_lbc
    else:
        individual_losses["lbc"] = torch.tensor(0.0, device=model_output_t1.device)

    # Combine losses using weights
    total_loss = torch.tensor(0.0, device=model_output_t1.device)
    if loss_weights.get("supervised", 0.0) > 0:
        total_loss += loss_weights["supervised"] * loss_supervised

    if loss_weights.get("divergence", 0.0) > 0:  # Continuity loss
        total_loss += loss_weights["divergence"] * loss_divergence

    if loss_weights.get("histogram", 0.0) > 0:
        total_loss += loss_weights["histogram"] * loss_histogram

    if use_ns_loss:  # Momentum part of N-S
        total_loss += loss_weights["navier_stokes"] * individual_losses["navier_stokes_momentum"]

    if use_lbc_loss:  # Boundary condition loss
        total_loss += loss_weights["lbc"] * individual_losses["lbc"]

    # Store weighted losses for logging if needed
    individual_losses["weighted_supervised"] = loss_weights.get("supervised", 0.0) * loss_supervised
    individual_losses["weighted_divergence"] = loss_weights.get("divergence", 0.0) * loss_divergence
    individual_losses["weighted_histogram"] = loss_weights.get("histogram", 0.0) * loss_histogram
    individual_losses["weighted_navier_stokes_momentum"] = loss_weights.get("navier_stokes", 0.0) * individual_losses[
        "navier_stokes_momentum"]
    individual_losses["weighted_lbc"] = loss_weights.get("lbc", 0.0) * individual_losses["lbc"]

    return total_loss, individual_losses


# --------------------------------------------------------------------- #
# Navier-Stokes Loss Components
# --------------------------------------------------------------------- #

def compute_temporal_derivative(
        u_pred_t1: torch.Tensor,  # Predicted velocity at t1 [N, 3]
        u_true_t0: torch.Tensor,  # True velocity at t0 [N, 3]
        dt: torch.Tensor | float  # Time step
) -> torch.Tensor:
    """Computes temporal derivative (u_pred_t1 - u_true_t0) / dt."""
    if isinstance(dt, torch.Tensor):
        dt = dt.to(u_pred_t1.device)
    # Ensure dt is not zero to avoid division errors
    if (isinstance(dt, float) and dt == 0.0) or \
            (isinstance(dt, torch.Tensor) and torch.any(dt == 0.0)):
        # Return zeros or raise error. For loss, zero contribution might be safer if this is rare.
        # However, dt=0 indicates a problem. For now, let's assume dt is valid.
        # If dt can be per-node and some are 0, that's an issue. Assume scalar dt or all non-zero.
        if isinstance(dt, torch.Tensor) and dt.numel() == 1:  # Make sure scalar tensor dt is float
            dt_val = dt.item()
            if dt_val == 0.0:
                raise ValueError("Time step dt is zero.")
        elif isinstance(dt, float) and dt == 0.0:
            raise ValueError("Time step dt is zero.")
        # If dt is a tensor with multiple values, ensure all are non-zero
        # This specific check might be overly cautious if dt is always scalar as expected

    dudt = (u_pred_t1 - u_true_t0.to(u_pred_t1.device)) / dt
    return dudt


# Placeholder for convective term (u · ∇)u
def compute_convective_term(
        velocity: torch.Tensor,  # Velocity field u, shape [N, 3]
        graph_data: Data  # Graph structure (pos, edge_index, edge_attr)
) -> torch.Tensor:
    """
    Computes the convective term (u · ∇)u for the Navier-Stokes equations.
    (u · ∇)u = u_j * ∂u_i / ∂x_j (summed over j, for each component i)
    Result is a vector of shape [N, 3].
    """
    grad_u = compute_vector_gradient(velocity, graph_data)  # Shape [N, 3, 3]

    # Unsqueeze velocity to be [N, 1, 3] for batch matrix multiplication
    # u_expanded = velocity.unsqueeze(1)

    # (u · ∇)u component-wise:
    # Result_i = sum_j u_j * (∂u_i / ∂x_j)
    # grad_u is [N, D_vel_comp, D_spatial_coord] where D_vel_comp is for u_i, D_spatial_coord is for x_j
    # velocity is [N, D_spatial_coord] (if u_j is velocity component along x_j)

    # Let velocity be u = (u0, u1, u2)
    # Let grad_u be G, where G[n, i, j] = ∂u_i / ∂x_j for node n.
    # We want c_i = sum_j u_j * (∂u_i / ∂x_j)
    # c_0 = u_0 * G_00 + u_1 * G_01 + u_2 * G_02
    # c_1 = u_0 * G_10 + u_1 * G_11 + u_2 * G_12
    # c_2 = u_0 * G_20 + u_1 * G_21 + u_2 * G_22
    # This is equivalent to (grad_u @ velocity.unsqueeze(-1)).squeeze(-1)
    # if grad_u is [N, D_out_component, D_in_component_of_u_for_mult]
    # Here, velocity components u_j multiply columns of ∇u_i.
    # This is sum(velocity_j * grad_u_ij for j in spatial_dims) for each component i of velocity.
    # Equivalent to einsum: 'nij,nj->ni' (N=nodes, i=velocity_component_out, j=spatial_dim/velocity_component_in)

    convective_term = torch.einsum('nij,nj->ni', grad_u, velocity)

    return convective_term


def compute_scalar_gradient(
        scalar_field: torch.Tensor,  # Scalar field p [N]
        graph_data: Data  # Graph structure (pos, edge_index, edge_attr)
) -> torch.Tensor:
    """
    Computes gradient of a scalar field on the graph.
    ∇f_i ≈ Σ_j ( (f_j - f_i) / ||pos_j - pos_i||² ) * (pos_j - pos_i)
    """
    num_nodes = scalar_field.size(0)
    edge_index = graph_data.edge_index
    # edge_attr should be pos[dst] - pos[src]
    edge_attr = graph_data.edge_attr.to(scalar_field.device)

    dist_sq = edge_attr.norm(dim=-1, p=2).pow(2).clamp(min=1e-12)  # Clamp for stability

    row, col = edge_index  # row=src, col=dst

    # Scalar difference f_dst - f_src
    df_edges = scalar_field[col] - scalar_field[row]

    # Weighted difference scaled by edge vector: ( (f_dst - f_src) / dist_sq ) * edge_attr
    # This represents the contribution of edge (src,dst) to the gradient at src (and related to dst)
    grad_terms = (df_edges / dist_sq).unsqueeze(-1) * edge_attr

    # Accumulate contributions at source nodes
    # For an edge (i,j), this term ( (f_j - f_i) / |r_ij|^2 * r_ij ) is added to grad_f_i
    # To get the gradient at node j from this edge, it would be ( (f_i - f_j) / |r_ji|^2 * r_ji )
    # which is ( (f_i - f_j) / |r_ij|^2 * (-r_ij) ). This is the negative of the term above.
    # So, if we sum `grad_terms` at `row` and `-grad_terms` at `col`, we should get a symmetric sum.
    # However, the standard formula sums (f_j - f_i) / |r_ij|^2 * r_ij for *all neighbors j of i*.
    # If the graph stores directed edges (i,j) and (j,i) for an undirected pair:
    #   - Edge (i,j): row=i, col=j. df=f_j-f_i. term = (f_j-f_i)/|r|^2 * r_ij. Add to node i.
    #   - Edge (j,i): row=j, col=i. df=f_i-f_j. term = (f_i-f_j)/|r|^2 * r_ji. Add to node j.
    # This seems correct if `scatter_add` is done onto `row`.

    grad_field = torch.zeros(num_nodes, 3, device=scalar_field.device)
    scatter_add(grad_terms, row, dim=0, out=grad_field)

    # Optional: Average by degree
    # degree = scatter_add(torch.ones_like(row, dtype=scalar_field.dtype), row, dim=0, dim_size=num_nodes).clamp(min=1).unsqueeze(-1)
    # grad_field = grad_field / degree

    return grad_field


def compute_pressure_gradient(
        pressure: torch.Tensor,  # Pressure field p [N]
        graph_data: Data  # Graph structure (pos, edge_index, edge_attr)
) -> torch.Tensor:
    """Computes pressure gradient ∇p using compute_scalar_gradient."""
    return compute_scalar_gradient(pressure, graph_data)


def compute_vector_gradient(
        vector_field: torch.Tensor,  # Vector field u [N, D_vec] (typically D_vec=3 for velocity)
        graph_data: Data  # Graph structure
) -> torch.Tensor:
    """
    Computes the gradient of a vector field, ∇u.
    Returns a tensor of shape [N, D_vec, D_spatial] (e.g., [N, 3, 3] for velocity).
    (∇u)_ij = ∂u_i / ∂x_j
    """
    num_nodes = vector_field.size(0)
    num_vector_dims = vector_field.size(1)  # Should be 3 for velocity
    num_spatial_dims = graph_data.pos.size(1)  # Should be 3 for 3D coordinates

    # Initialize gradient tensor: grad_u[node_idx, component_idx, spatial_dim_idx]
    grad_vector = torch.zeros(num_nodes, num_vector_dims, num_spatial_dims, device=vector_field.device)

    for i in range(num_vector_dims):
        # Compute gradient for the i-th component of the vector field
        scalar_component = vector_field[:, i]
        grad_scalar_component = compute_scalar_gradient(scalar_component, graph_data)  # Result is [N, D_spatial]
        grad_vector[:, i, :] = grad_scalar_component

    return grad_vector


# Placeholder for Laplacian term ∇²u
def compute_laplacian_term(
        field: torch.Tensor,  # Scalar or Vector field u [N] or [N,D]
        graph_data: Data  # Graph structure
) -> torch.Tensor:
    """
    Computes Laplacian of a field (scalar or vector) on the graph.
    ∇²f_i ≈ Σ_j (f_j - f_i) / ||pos_j - pos_i||²  (sum over neighbors j of i)
    If f is a vector, applies component-wise.
    """
    num_nodes = field.size(0)
    edge_index = graph_data.edge_index
    pos = graph_data.pos.to(field.device)
    edge_attr = graph_data.edge_attr.to(field.device)  # pos[dst] - pos[src]

    dist_sq = edge_attr.norm(dim=-1, p=2).pow(2).clamp(min=1e-12)
    inv_dist_sq = 1.0 / dist_sq

    row, col = edge_index  # row=src, col=dst

    # Difference: field_col - field_row (i.e. field_neighbor - field_node)
    # This is (f_j - f_i) for an edge (i,j)
    field_diff = field[col] - field[row]  # Shape [num_edges, num_features] or [num_edges]

    # Weighted difference: (f_j - f_i) / ||dist||^2
    weighted_diff = field_diff * inv_dist_sq.unsqueeze(-1)  # Works if field_diff is [E,D] or [E]

    # Sum these contributions at node `row` (i)
    laplacian_val = torch.zeros_like(field)
    scatter_add(weighted_diff, row, dim=0, out=laplacian_val)

    # The formulation sum (f_j - f_i) implies for node i, we sum contributions from all neighbors j.
    # The current scatter_add sums at `row` (source node of directed edge).
    # If graph is undirected (both (i,j) and (j,i) exist), this will capture all neighbors.

    return laplacian_val


def navier_stokes_loss(
        predicted_output_t1: torch.Tensor,  # [N, 4] (vx, vy, vz, p) at t1
        graph_t0: Data,  # Data object at t0 (contains true_vel_t0, pos, time)
        graph_t1: Data,  # Data object at t1 (contains time, and pos if mesh moves)
        reynolds_number: float,
        reduction: str = 'mean'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Navier-Stokes equation residuals as a loss.
    Assumes inputs (velocity, pressure, coordinates, time) are non-dimensional.
    Equation: ∂u/∂t + (u·∇)u + ∇p - (1/Re)∇²u = 0 (Momentum)
              ∇·u = 0 (Continuity)

    Args:
        predicted_output_t1: Model prediction at t1, shape [N, 4], where last dim is pressure.
        graph_t0: Graph data at t0. `graph_t0.x` is u_true_t0. `graph_t0.time`.
        graph_t1: Graph data at t1. `graph_t1.time`. `graph_t1.pos` for spatial ops at t1.
                  (Using graph_t1 for spatial attributes like pos, edge_index for derivatives at t1)
        reynolds_number: The Reynolds number (dimensionless).
        reduction: 'mean' or 'sum' for the final loss.

    Returns:
        A tuple: (total_ns_loss, momentum_loss_mean_sq, continuity_loss_mean_sq)
    """
    u_pred_t1 = predicted_output_t1[:, :3]
    p_pred_t1 = predicted_output_t1[:, 3]  # Assuming pressure is the 4th component

    u_true_t0 = graph_t0.x.to(u_pred_t1.device)  # Velocity at previous time step

    dt = graph_t1.time.item() - graph_t0.time.item()  # scalar
    if dt <= 1e-9:  # Avoid division by zero or very small dt
        # This might happen if t0 and t1 frames are identical or time is not progressing.
        # Return a zero loss or handle as an error. For now, zero loss contribution.
        print(f"Warning: dt is very small or zero ({dt}). Navier-Stokes loss might be unreliable.")
        # Fallback to a nominal dt to prevent NaN/inf, but this isn't ideal.
        # A better solution would be to ensure valid dt from data pipeline.
        if dt == 0.0: dt = 1e-6  # Avoid direct zero division if absolutely necessary for code flow

    # --- Calculate terms for Momentum Equation ---
    # 1. Temporal derivative: ∂u/∂t
    dudt = compute_temporal_derivative(u_pred_t1, u_true_t0, dt)

    # For spatial derivatives, use graph_t1 as it represents the state at t1
    # Ensure graph_t1 has necessary attributes (pos, edge_index, edge_attr)
    # If graph_t0 and graph_t1 share structure, graph_t0 could be used too.
    # Assuming graph_t1.pos, graph_t1.edge_index etc. are what we need for derivatives at t1.
    current_graph_structure = graph_t1  # Or graph_t0 if structure is static

    # 2. Convective term: (u·∇)u based on u_pred_t1
    conv_term = compute_convective_term(u_pred_t1, current_graph_structure)

    # 3. Pressure gradient: ∇p based on p_pred_t1
    grad_p = compute_pressure_gradient(p_pred_t1, current_graph_structure)

    # 4. Viscous term: (1/Re)∇²u based on u_pred_t1
    laplacian_u = compute_laplacian_term(u_pred_t1, current_graph_structure)
    viscous_term = (1.0 / reynolds_number) * laplacian_u

    # Momentum equation residual (target is 0)
    # Res_mom = ∂u/∂t + (u·∇)u + ∇p - (1/Re)∇²u
    momentum_residual = dudt + conv_term + grad_p - viscous_term

    # --- Calculate term for Continuity Equation ---
    # ∇·u based on u_pred_t1
    div_u = calculate_divergence(u_pred_t1, current_graph_structure)  # div_u is [N]

    # --- Calculate losses (typically mean squared error of residuals) ---
    momentum_loss = momentum_residual.pow(2).sum(dim=1)  # Sum over 3 components, then mean over N
    continuity_loss = div_u.pow(2)  # Already [N]

    if reduction == 'mean':
        final_momentum_loss = momentum_loss.mean()
        final_continuity_loss = continuity_loss.mean()
    elif reduction == 'sum':
        final_momentum_loss = momentum_loss.sum()
        final_continuity_loss = continuity_loss.sum()
    else:
        raise ValueError(f"Unknown reduction type: {reduction}")

    total_loss = final_momentum_loss + final_continuity_loss

    return total_loss, final_momentum_loss, final_continuity_loss


# --------------------------------------------------------------------- #
# Boundary Condition Loss
# --------------------------------------------------------------------- #

def boundary_condition_loss(
        predicted_velocity_t1: torch.Tensor,  # Predicted velocity at t1 [N, 3]
        boundary_nodes_mask: torch.Tensor | None,  # Boolean tensor [N], True for boundary nodes
        target_boundary_velocity: torch.Tensor | float = 0.0,
        # Target velocity for boundary nodes [N_boundary, 3] or scalar
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Calculates a loss based on boundary conditions for velocity.
    Default is no-slip (u=0) if target_boundary_velocity is 0.0.

    Args:
        predicted_velocity_t1: Predicted velocity tensor [num_nodes, 3].
        boundary_nodes_mask: Boolean tensor indicating boundary nodes. If None or all False, loss is 0.
        target_boundary_velocity: Target velocity for boundary nodes.
                                  Can be a scalar (e.g., 0.0 for no-slip) or a tensor of shape
                                  [num_boundary_nodes, 3] for specific velocities.
        reduction: 'mean', 'sum'.

    Returns:
        Scalar loss tensor.
    """
    if boundary_nodes_mask is None or not torch.any(boundary_nodes_mask):
        return torch.tensor(0.0, device=predicted_velocity_t1.device)

    pred_boundary_vel = predicted_velocity_t1[boundary_nodes_mask]

    if pred_boundary_vel.numel() == 0:  # No boundary nodes selected by mask actually existed
        return torch.tensor(0.0, device=predicted_velocity_t1.device)

    if isinstance(target_boundary_velocity, float):
        # If scalar, create a zero tensor of the same shape as pred_boundary_vel
        target_vel = torch.full_like(pred_boundary_vel, target_boundary_velocity)
    else:
        target_vel = target_boundary_velocity.to(pred_boundary_vel.device)
        if target_vel.shape != pred_boundary_vel.shape:
            # This might happen if target_boundary_velocity was [N_boundary_true_count, 3]
            # but boundary_nodes_mask selected a different number or order.
            # For simplicity, assume target_vel should match pred_boundary_vel if provided as tensor.
            # A more robust way would be to ensure target_boundary_velocity is indexed by the same mask
            # or is already filtered.
            # Current assumption: if tensor, it's already correctly filtered and shaped.
            # This might need adjustment based on how boundary_nodes_mask and target_boundary_velocity are supplied.
            # For now, let's assume it's either a scalar or a correctly pre-filtered tensor matching pred_boundary_vel.
            raise ValueError(
                f"Shape mismatch for target_boundary_velocity. Expected {pred_boundary_vel.shape}, got {target_vel.shape}."
                " If providing a tensor, ensure it corresponds to the nodes selected by boundary_nodes_mask."
            )

    loss = F.mse_loss(pred_boundary_vel, target_vel, reduction='none').sum(dim=-1)  # Sum MSE over velocity components

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # Should not happen if we stick to 'mean' or 'sum'
        raise ValueError(f"Unknown reduction type for LBC: {reduction}")


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
    edge_attr_test = pos[edge_idx[1]] - pos[edge_idx[0]]  # Target - Source

    test_graph_data = Data(
        x=true_vel,  # Not used by divergence directly, but part of graph
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

    # --- Test Navier-Stokes Loss ---
    print("\nTesting navier_stokes_loss...")
    num_nodes_ns = 10
    # Predicted output at t1 (vel_x, vel_y, vel_z, pressure)
    pred_output_t1_ns = torch.randn(num_nodes_ns, 4, device=device, requires_grad=True)

    # Graph data for t0
    u_true_t0_ns = torch.randn(num_nodes_ns, 3, device=device)
    pos_t0_ns = torch.rand(num_nodes_ns, 3, device=device) * 0.1  # Small domain for t0
    edge_index_t0_ns = torch.randint(0, num_nodes_ns, (2, num_nodes_ns * 3), device=device)  # Dummy edges
    edge_attr_t0_ns = pos_t0_ns[edge_index_t0_ns[1]] - pos_t0_ns[edge_index_t0_ns[0]]
    graph_t0_ns = Data(x=u_true_t0_ns, pos=pos_t0_ns, edge_index=edge_index_t0_ns, edge_attr=edge_attr_t0_ns,
                       time=torch.tensor([0.0], device=device))

    # Graph data for t1 (spatial structure for derivatives at t1)
    pos_t1_ns = pos_t0_ns + torch.rand(num_nodes_ns, 3, device=device) * 0.01  # Slightly moved mesh
    edge_index_t1_ns = edge_index_t0_ns  # Assume same connectivity for simplicity
    edge_attr_t1_ns = pos_t1_ns[edge_index_t1_ns[1]] - pos_t1_ns[edge_index_t1_ns[0]]
    graph_t1_ns = Data(pos=pos_t1_ns, edge_index=edge_index_t1_ns, edge_attr=edge_attr_t1_ns,
                       time=torch.tensor([0.01], device=device))

    re_ns = 100.0

    # Clear gradients before new backward pass if any
    if pred_output_t1_ns.grad is not None:
        pred_output_t1_ns.grad.zero_()

    ns_total_loss, ns_mom_loss, ns_cont_loss = navier_stokes_loss(
        pred_output_t1_ns, graph_t0_ns, graph_t1_ns, re_ns, reduction='mean'
    )
    print(f"Navier-Stokes Total Loss: {ns_total_loss.item()}")
    print(f"Navier-Stokes Momentum Loss: {ns_mom_loss.item()}")
    print(f"Navier-Stokes Continuity Loss: {ns_cont_loss.item()}")

    assert ns_total_loss.ndim == 0
    assert ns_mom_loss.ndim == 0
    assert ns_cont_loss.ndim == 0

    try:
        ns_total_loss.backward()
        print("Navier-Stokes loss backward pass successful.")
        assert pred_output_t1_ns.grad is not None, "Gradients not computed for N-S loss."
        # print(f"Sample gradient for pred_output_t1_ns: {pred_output_t1_ns.grad[0].tolist()}")
    except Exception as e:
        print(f"Error during N-S loss backward pass: {e}")
        raise

    # --- Test combined_loss with N-S ---
    print("\nTesting combined_loss with Navier-Stokes...")
    loss_weights_ns = {
        "supervised": 1.0,
        "divergence": 1.0,  # Continuity from N-S
        "navier_stokes": 1.0,  # Momentum from N-S
        "histogram": 0.0  # Off for this test
    }
    # For combined_loss, true_velocity_t1 is needed for supervised loss
    true_vel_t1_ns = torch.randn(num_nodes_ns, 3, device=device)

    if pred_output_t1_ns.grad is not None:
        pred_output_t1_ns.grad.zero_()

    total_loss_combined_ns, individual_losses_ns = combined_loss(
        model_output_t1=pred_output_t1_ns,
        true_velocity_t1=true_vel_t1_ns,
        graph_t0=graph_t0_ns,
        graph_t1=graph_t1_ns,
        loss_weights=loss_weights_ns,
        reynolds_number=re_ns,
        histogram_bins=32
    )
    print(f"Combined Total Loss (with N-S): {total_loss_combined_ns.item()}")
    print(f"Individual Losses (with N-S):")
    for k, v_tensor in individual_losses_ns.items():
        print(f"  {k}: {v_tensor.item() if isinstance(v_tensor, torch.Tensor) else v_tensor}")

    assert "navier_stokes_momentum" in individual_losses_ns
    assert "divergence" in individual_losses_ns  # Continuity

    try:
        total_loss_combined_ns.backward()
        print("Combined loss (with N-S) backward pass successful.")
        assert pred_output_t1_ns.grad is not None, "Gradients not computed for combined N-S loss."
    except Exception as e:
        print(f"Error during combined N-S loss backward pass: {e}")
        raise

    # --- Test Boundary Condition Loss ---
    print("\nTesting boundary_condition_loss...")
    num_nodes_bc = 10
    pred_vel_bc = torch.randn(num_nodes_bc, 3, device=device, requires_grad=True)

    # Test 1: No mask
    lbc_no_mask = boundary_condition_loss(pred_vel_bc, None, 0.0)
    print(f"LBC with no mask: {lbc_no_mask.item()}")
    assert torch.isclose(lbc_no_mask, torch.tensor(0.0))

    # Test 2: Mask with no True values
    mask_all_false = torch.zeros(num_nodes_bc, dtype=torch.bool, device=device)
    lbc_all_false = boundary_condition_loss(pred_vel_bc, mask_all_false, 0.0)
    print(f"LBC with all_false mask: {lbc_all_false.item()}")
    assert torch.isclose(lbc_all_false, torch.tensor(0.0))

    # Test 3: Mask selects some nodes, target is scalar 0.0 (no-slip)
    mask_some_true = torch.tensor([True, False, True] + [False] * (num_nodes_bc - 3), dtype=torch.bool, device=device)
    num_boundary_nodes = mask_some_true.sum().item()

    lbc_some_scalar_target = boundary_condition_loss(pred_vel_bc, mask_some_true, 0.0, reduction='mean')
    expected_lbc_val = F.mse_loss(pred_vel_bc[mask_some_true], torch.zeros_like(pred_vel_bc[mask_some_true])).item()
    # Note: boundary_condition_loss internal MSE is sum over components, then mean over nodes.
    # F.mse_loss default is mean over all elements. So, need to be careful with direct comparison.
    # Manual calculation for this specific case:
    expected_manual_lbc = (pred_vel_bc[mask_some_true].pow(2).sum(dim=-1)).mean().item()
    print(
        f"LBC some nodes, scalar target 0.0: {lbc_some_scalar_target.item()} (Expected approx: {expected_manual_lbc})")
    assert torch.isclose(lbc_some_scalar_target, torch.tensor(expected_manual_lbc, device=device))

    # Test 4: Mask selects some nodes, target is a tensor
    target_vel_bc_tensor = torch.randn(num_boundary_nodes, 3, device=device)
    # This requires target_boundary_velocity to be pre-filtered or shaped correctly.
    # The current LBC function expects target_vel to match pred_boundary_vel if tensor.
    # For this test, let's filter the target_vel based on the mask for the LBC function.
    # However, the LBC function itself now expects the passed tensor to be already filtered.
    # So, we create a target that would match after filtering.

    # To test the LBC function correctly when target is a tensor:
    # We need to simulate how it would be used: target_boundary_velocity is for the *actual* boundary nodes.
    # The LBC function will then select from predicted_velocity_t1 using the mask.
    # Let's make target_boundary_velocity have the same shape as pred_vel_bc[mask_some_true]

    lbc_tensor_target = boundary_condition_loss(pred_vel_bc, mask_some_true, target_vel_bc_tensor, reduction='sum')
    expected_lbc_tensor_sum = F.mse_loss(pred_vel_bc[mask_some_true], target_vel_bc_tensor,
                                         reduction='none').sum().item()
    # sum(dim=-1).sum() for our LBC's sum reduction
    expected_manual_lbc_tensor_sum = (
        (pred_vel_bc[mask_some_true] - target_vel_bc_tensor).pow(2).sum(dim=-1)).sum().item()
    print(
        f"LBC some nodes, tensor target (sum): {lbc_tensor_target.item()} (Expected approx: {expected_manual_lbc_tensor_sum})")
    assert torch.isclose(lbc_tensor_target, torch.tensor(expected_manual_lbc_tensor_sum, device=device))

    lbc_tensor_target.backward()  # Test backward pass
    print("LBC backward pass successful.")
    assert pred_vel_bc.grad is not None

    # --- Test combined_loss with LBC ---
    print("\nTesting combined_loss with LBC...")
    # Reusing some data from N-S test for graph_t0, graph_t1
    # model_output_t1 will be pred_output_t1_ns
    # true_velocity_t1 will be true_vel_t1_ns

    loss_weights_lbc = {
        "supervised": 0.5,
        "divergence": 0.0,
        "navier_stokes": 0.0,
        "histogram": 0.0,
        "lbc": 1.0
    }
    # Create a boundary mask for graph_t1_ns (which has num_nodes_ns nodes)
    bc_mask_combined = torch.zeros(num_nodes_ns, dtype=torch.bool, device=device)
    if num_nodes_ns >= 2:
        bc_mask_combined[0] = True
        bc_mask_combined[-1] = True

    # Target for these boundary nodes (scalar 0.0 for no-slip)
    target_bc_vel_combined = 0.0

    # Ensure pred_output_t1_ns's grad is cleared
    if pred_output_t1_ns.grad is not None:
        pred_output_t1_ns.grad.zero_()

    total_loss_combined_lbc, individual_losses_lbc = combined_loss(
        model_output_t1=pred_output_t1_ns,  # Shape [num_nodes_ns, 4]
        true_velocity_t1=true_vel_t1_ns,  # Shape [num_nodes_ns, 3]
        graph_t0=graph_t0_ns,
        graph_t1=graph_t1_ns,  # Can store boundary_mask here if needed, or pass separately
        loss_weights=loss_weights_lbc,
        reynolds_number=re_ns,  # Not used if navier_stokes weight is 0
        boundary_nodes_mask=bc_mask_combined,
        target_boundary_velocity=target_bc_vel_combined
    )
    print(f"Combined Total Loss (with LBC): {total_loss_combined_lbc.item()}")
    print(f"Individual Losses (with LBC):")
    for k, v_tensor in individual_losses_lbc.items():
        print(f"  {k}: {v_tensor.item() if isinstance(v_tensor, torch.Tensor) else v_tensor}")

    assert "lbc" in individual_losses_lbc
    assert individual_losses_lbc["lbc"].item() > 0 if bc_mask_combined.sum() > 0 else True

    try:
        total_loss_combined_lbc.backward()
        print("Combined loss (with LBC) backward pass successful.")
        assert pred_output_t1_ns.grad is not None
    except Exception as e:
        print(f"Error during combined LBC loss backward pass: {e}")
        raise

    print("\nlosses.py tests passed.")
