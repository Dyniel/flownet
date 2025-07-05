# -*- coding: utf-8 -*-
"""
models.py
---------
Definitions for GNN models (FlowNet, RotFlowNet) and their components (MLP, GNNStep).
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint

# Ensure this MLP is flexible enough for various uses (encoder, decoder, within GNNStep)
def MLP(
    in_features: int,
    out_features: int,
    hidden_features: int = 128,
    num_layers: int = 2, # Total layers including output; 1 means Linear, 2 means Linear-ReLU-Linear
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module | None = None,
    checkpoint_intermediate_layers: bool = False # This flag is a signal for BaseFlowGNN
):
    """
    Creates a Multi-Layer Perceptron (MLP).
    The checkpoint_intermediate_layers flag is not used inside this MLP factory itself,
    but is passed to BaseFlowGNN to indicate that its layers should be checkpointed individually.
    """
    layers = []
    current_features = in_features

    if num_layers == 1:
        layers.append(nn.Linear(current_features, out_features))
    else:
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_features, hidden_features))
            layers.append(activation())
            current_features = hidden_features
        layers.append(nn.Linear(current_features, out_features))

    if output_activation:
        layers.append(output_activation())

    return nn.Sequential(*layers)


class GNNStep(nn.Module):
    """
    A single step/layer in the GNN, performing edge and node updates.
    """
    def __init__(self, hidden_dim: int, edge_mlp_layers: int = 2, node_mlp_layers: int = 2, activation=nn.ReLU):
        super().__init__()
        self.edge_mlp = MLP(
            in_features=3 * hidden_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            num_layers=edge_mlp_layers,
            activation=activation
        )
        self.node_mlp = MLP(
            in_features=2 * hidden_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            num_layers=node_mlp_layers,
            activation=activation
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        edge_inputs = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.edge_mlp(edge_inputs)
        aggregated_messages = scatter_add(messages, col, dim=0, dim_size=x.size(0))
        node_inputs = torch.cat([x, aggregated_messages], dim=-1)
        new_x = self.node_mlp(node_inputs)
        return new_x


class BaseFlowGNN(nn.Module):
    """
    Base GNN architecture for flow prediction.
    """
    def __init__(
        self,
        node_in_features: int = 3,
        edge_in_features: int = 3,
        node_out_features: int = 3,
        hidden_dim: int = 128,
        num_gnn_layers: int = 5,
        encoder_mlp_layers: int = 2,
        decoder_mlp_layers: int = 2,
        gnn_step_mlp_layers: int = 2,
        activation_fn=nn.ReLU,
        checkpoint_edge_encoder_internals: bool = False # Flag for granular checkpointing
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Store the flag for use in the forward pass
        self.checkpoint_edge_encoder_internals = checkpoint_edge_encoder_internals

        self.node_encoder = MLP(
            node_in_features, hidden_dim, hidden_dim, encoder_mlp_layers, activation=activation_fn,
            checkpoint_intermediate_layers=False # node_encoder typically not checkpointed internally
        )
        self.edge_encoder = MLP(
            edge_in_features, hidden_dim, hidden_dim, encoder_mlp_layers, activation=activation_fn,
            # The MLP factory receives the flag, but doesn't act on it.
            # BaseFlowGNN's forward pass will use self.checkpoint_edge_encoder_internals
            checkpoint_intermediate_layers=checkpoint_edge_encoder_internals
        )

        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(GNNStep(hidden_dim, gnn_step_mlp_layers, gnn_step_mlp_layers, activation=activation_fn))

        self.node_decoder = MLP(
            hidden_dim, node_out_features, hidden_dim, decoder_mlp_layers, activation=activation_fn,
            output_activation=None
        )

    def forward(self, data: Data) -> torch.Tensor:
        h_node = self.node_encoder(data.x)

        # Process edge features with optional granular checkpointing
        if self.checkpoint_edge_encoder_internals:
            # Apply checkpointing to each layer within the edge_encoder MLP
            # self.edge_encoder is an nn.Sequential
            temp_h_edge = data.edge_attr
            if not temp_h_edge.requires_grad and torch.is_grad_enabled():
                 # Ensure inputs to checkpointed segments that require grad have requires_grad=True
                 # This is often needed if data.edge_attr is a leaf tensor from data loader.
                 # Only set if grads are enabled for the overall computation.
                temp_h_edge.requires_grad_(True)

            for layer in self.edge_encoder:
                # Preserve RNG state for checkpoint is important if layers have stochasticity (e.g. Dropout)
                # and non_reentrant version is used. For Linear/ReLU, it's less critical but good practice.
                temp_h_edge = checkpoint(layer, temp_h_edge, use_reentrant=False, preserve_rng_state=True)
            h_edge = temp_h_edge
        else:
            # Original behavior or module-level checkpoint (if added back)
            h_edge = self.edge_encoder(data.edge_attr)

        for conv_layer in self.convs:
            # Checkpointing GNN steps
            if not h_node.requires_grad and torch.is_grad_enabled(): h_node.requires_grad_(True)
            if not h_edge.requires_grad and torch.is_grad_enabled(): h_edge.requires_grad_(True)
            h_node = checkpoint(conv_layer, h_node, data.edge_index, h_edge, use_reentrant=False, preserve_rng_state=True)

        predictions = self.node_decoder(h_node)
        return predictions

class FlowNet(BaseFlowGNN):
    def __init__(self, cfg: dict):
        super().__init__(
            node_in_features=cfg.get("node_in_features", 3),
            edge_in_features=cfg.get("edge_in_features", 3),
            node_out_features=cfg.get("node_out_features", 3),
            hidden_dim=cfg.get("h_dim", 128),
            num_gnn_layers=cfg.get("layers", 5),
            encoder_mlp_layers=cfg.get("encoder_mlp_layers", 2),
            decoder_mlp_layers=cfg.get("decoder_mlp_layers", 2),
            gnn_step_mlp_layers=cfg.get("gnn_step_mlp_layers", 2),
            checkpoint_edge_encoder_internals=cfg.get("checkpoint_edge_encoder_internals", False)
        )

class RotFlowNet(BaseFlowGNN):
    def __init__(self, cfg: dict):
        super().__init__(
            node_in_features=cfg.get("node_in_features", 3),
            edge_in_features=cfg.get("edge_in_features", 3),
            node_out_features=cfg.get("node_out_features", 3),
            hidden_dim=cfg.get("h_dim", 128),
            num_gnn_layers=cfg.get("layers", 5),
            encoder_mlp_layers=cfg.get("encoder_mlp_layers", 2),
            decoder_mlp_layers=cfg.get("decoder_mlp_layers", 2),
            gnn_step_mlp_layers=cfg.get("gnn_step_mlp_layers", 2),
            checkpoint_edge_encoder_internals=cfg.get("checkpoint_edge_encoder_internals", False)
        )

if __name__ == '__main__':
    print("Testing models.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_params = {
        "node_in_features": 3, "edge_in_features": 3, "node_out_features": 3,
        "h_dim": 64, "layers": 2, "encoder_mlp_layers": 2, "decoder_mlp_layers": 2,
        "gnn_step_mlp_layers": 2, "checkpoint_edge_encoder_internals": True # Test granular checkpointing
    }
    num_nodes, num_edges = 10, 20
    dummy_x = torch.randn(num_nodes, config_params["node_in_features"], device=device)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    # Make edge_attr require grad for testing checkpointing if it's a leaf
    dummy_edge_attr = torch.randn(num_edges, config_params["edge_in_features"], device=device)
    dummy_pos = torch.randn(num_nodes, 3, device=device)
    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, pos=dummy_pos).to(device)

    flownet_model = FlowNet(config_params).to(device)
    print("\nFlowNet Model (with granular checkpointing for edge_encoder if enabled by config):")
    print(flownet_model)
    try:
        # Ensure model is in training mode for checkpointing effects if they differ
        flownet_model.train()
        output_flownet = flownet_model(dummy_data)
        print(f"FlowNet output shape: {output_flownet.shape}")
        # Dummy backward pass to check if checkpointing recomputation works
        output_flownet.sum().backward()
        print("FlowNet forward and backward pass with granular checkpointing successful.")
    except Exception as e:
        print(f"Error during FlowNet forward/backward pass with granular checkpointing: {e}")
        raise
    print("\nmodels.py tests passed.")
