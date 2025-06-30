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

# Ensure this MLP is flexible enough for various uses (encoder, decoder, within GNNStep)
def MLP(
    in_features: int,
    out_features: int,
    hidden_features: int = 128,
    num_layers: int = 2, # Total layers including output; 1 means Linear, 2 means Linear-ReLU-Linear
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module | None = None
):
    """
    Creates a Multi-Layer Perceptron (MLP).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_features: Number of features in hidden layers.
        num_layers: Total number of linear layers. If 1, it's a single linear layer.
        activation: Activation function to use for hidden layers.
        output_activation: Optional activation function for the output layer.
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
        # Edge MLP: processes [node_feat_src, node_feat_dst, edge_attr_feat]
        # Assuming node features and edge attributes are all of size hidden_dim after initial encoding
        self.edge_mlp = MLP(
            in_features=3 * hidden_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim, # Or allow separate hidden_dim_mlp
            num_layers=edge_mlp_layers,
            activation=activation
        )
        # Node MLP: processes [node_feat_current, aggregated_edge_messages]
        self.node_mlp = MLP(
            in_features=2 * hidden_dim,
            out_features=hidden_dim,
            hidden_features=hidden_dim,
            num_layers=node_mlp_layers,
            activation=activation
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features, shape [num_nodes, hidden_dim].
            edge_index: Edge connectivity, shape [2, num_edges].
            edge_attr: Edge attributes, shape [num_edges, hidden_dim] (after encoding).

        Returns:
            Updated node features, shape [num_nodes, hidden_dim].
        """
        row, col = edge_index  # source_nodes, target_nodes

        # 1. Message construction (on edges)
        # Concatenate source node features, target node features, and edge attributes
        edge_inputs = torch.cat([x[row], x[col], edge_attr], dim=-1) # [num_edges, 3 * hidden_dim]
        messages = self.edge_mlp(edge_inputs)                     # [num_edges, hidden_dim]

        # 2. Message aggregation (to target nodes)
        # Sum messages for each target node
        aggregated_messages = scatter_add(messages, col, dim=0, dim_size=x.size(0)) # [num_nodes, hidden_dim]

        # 3. Node update
        # Concatenate current node features with aggregated messages
        node_inputs = torch.cat([x, aggregated_messages], dim=-1) # [num_nodes, 2 * hidden_dim]
        new_x = self.node_mlp(node_inputs)                      # [num_nodes, hidden_dim]

        return new_x


class BaseFlowGNN(nn.Module):
    """
    Base GNN architecture for flow prediction.
    Can be specialized into FlowNet or RotFlowNet by specific graph construction.
    """
    def __init__(
        self,
        node_in_features: int = 3,    # e.g., 3 for velocity U
        edge_in_features: int = 3,    # e.g., 3 for relative position
        node_out_features: int = 3,   # e.g., 3 for predicted velocity U'
        hidden_dim: int = 128,
        num_gnn_layers: int = 5,
        encoder_mlp_layers: int = 2,
        decoder_mlp_layers: int = 2, # Can be 2 for Linear-ReLU-Linear or 1 for Linear
        gnn_step_mlp_layers: int = 2,
        activation_fn=nn.ReLU
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder for node features (e.g., velocity)
        self.node_encoder = MLP(
            node_in_features, hidden_dim, hidden_dim, encoder_mlp_layers, activation=activation_fn
        )
        # Encoder for edge features (e.g., relative positions)
        self.edge_encoder = MLP(
            edge_in_features, hidden_dim, hidden_dim, encoder_mlp_layers, activation=activation_fn
        )

        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(GNNStep(hidden_dim, gnn_step_mlp_layers, gnn_step_mlp_layers, activation=activation_fn))

        # Decoder for node features (to predict output, e.g., new velocity)
        self.node_decoder = MLP(
            hidden_dim, node_out_features, hidden_dim, decoder_mlp_layers, activation=activation_fn,
            output_activation=None # Usually no activation on final regression output
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyTorch Geometric Data object with attributes:
                  - x: Node features (e.g., velocity), shape [num_nodes, node_in_features]
                  - edge_index: Edge connectivity, shape [2, num_edges]
                  - edge_attr: Edge attributes (e.g., relative positions), shape [num_edges, edge_in_features]
                  - pos: Node positions (optional, but often present), shape [num_nodes, 3]

        Returns:
            Predicted node features (e.g., velocity), shape [num_nodes, node_out_features]
        """
        # 1. Encode node and edge features
        h_node = self.node_encoder(data.x)        # [num_nodes, hidden_dim]
        h_edge = self.edge_encoder(data.edge_attr) # [num_edges, hidden_dim]

        # 2. Message passing through GNN layers
        for conv_layer in self.convs:
            h_node = conv_layer(h_node, data.edge_index, h_edge)
            # Potentially add residual connections or normalization here if needed

        # 3. Decode node features to get predictions
        predictions = self.node_decoder(h_node) # [num_nodes, node_out_features]

        return predictions

# Specific aliases or minor variations if needed
class FlowNet(BaseFlowGNN):
    """
    Standard FlowNet model.
    Inherits from BaseFlowGNN. Graph construction (kNN, full mesh) is handled externally.
    """
    def __init__(self, cfg: dict): # cfg is a dictionary with model parameters
        super().__init__(
            node_in_features=cfg.get("node_in_features", 3),
            edge_in_features=cfg.get("edge_in_features", 3),
            node_out_features=cfg.get("node_out_features", 3),
            hidden_dim=cfg.get("h_dim", 128),
            num_gnn_layers=cfg.get("layers", 5),
            encoder_mlp_layers=cfg.get("encoder_mlp_layers", 2),
            decoder_mlp_layers=cfg.get("decoder_mlp_layers", 2),
            gnn_step_mlp_layers=cfg.get("gnn_step_mlp_layers", 2),
            # activation_fn can also be configured if needed
        )

class RotFlowNet(BaseFlowGNN):
    """
    RotFlowNet model, potentially using different graph features (e.g., cylindrical coordinates)
    handled during graph construction time. The core architecture is the same as BaseFlowGNN.
    """
    def __init__(self, cfg: dict): # cfg is a dictionary with model parameters
        super().__init__(
            node_in_features=cfg.get("node_in_features", 3),
            edge_in_features=cfg.get("edge_in_features", 3),
            node_out_features=cfg.get("node_out_features", 3),
            hidden_dim=cfg.get("h_dim", 128),
            num_gnn_layers=cfg.get("layers", 5),
            encoder_mlp_layers=cfg.get("encoder_mlp_layers", 2),
            decoder_mlp_layers=cfg.get("decoder_mlp_layers", 2),
            gnn_step_mlp_layers=cfg.get("gnn_step_mlp_layers", 2),
        )


if __name__ == '__main__':
    # Example Usage & Test
    print("Testing models.py...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy configuration
    config_params = {
        "node_in_features": 3,
        "edge_in_features": 3,
        "node_out_features": 3,
        "h_dim": 64, # Smaller for quick test
        "layers": 2, # Fewer layers for quick test
        "encoder_mlp_layers": 2,
        "decoder_mlp_layers": 2,
        "gnn_step_mlp_layers": 2
    }

    # Create a dummy graph
    num_nodes = 10
    num_edges = 20
    dummy_x = torch.randn(num_nodes, config_params["node_in_features"], device=device)
    dummy_edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    dummy_edge_attr = torch.randn(num_edges, config_params["edge_in_features"], device=device)
    dummy_pos = torch.randn(num_nodes, 3, device=device)

    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, edge_attr=dummy_edge_attr, pos=dummy_pos).to(device)

    # Test FlowNet
    flownet_model = FlowNet(config_params).to(device)
    print("\nFlowNet Model:")
    print(flownet_model)
    try:
        output_flownet = flownet_model(dummy_data)
        print(f"FlowNet output shape: {output_flownet.shape} (Expected: [{num_nodes}, {config_params['node_out_features']}])")
        assert output_flownet.shape == (num_nodes, config_params["node_out_features"])
    except Exception as e:
        print(f"Error during FlowNet forward pass: {e}")
        raise

    # Test RotFlowNet
    rotflownet_model = RotFlowNet(config_params).to(device)
    print("\nRotFlowNet Model:")
    print(rotflownet_model)
    try:
        output_rotflownet = rotflownet_model(dummy_data)
        print(f"RotFlowNet output shape: {output_rotflownet.shape} (Expected: [{num_nodes}, {config_params['node_out_features']}])")
        assert output_rotflownet.shape == (num_nodes, config_params["node_out_features"])
    except Exception as e:
        print(f"Error during RotFlowNet forward pass: {e}")
        raise

    print("\nmodels.py tests passed.")
