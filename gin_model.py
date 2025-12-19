"""
GIN (Graph Isomorphism Network) for Link Prediction.

Implementation of GIN encoder and decoder for edge prediction
on semantic graphs according to the paper:
"How Powerful are Graph Neural Networks?" (Xu et al., ICLR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GINConvLayer(nn.Module):
    """
    Single GIN convolution layer.
    
    Aggregation formula:
        h_v^(k) = MLP^(k)((1 + eps) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1))
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        eps: Initial value of epsilon
        train_eps: If True, epsilon is learnable
    """
    
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0, train_eps: bool = True):
        super(GINConvLayer, self).__init__()
        
        # MLP for transforming aggregated features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        
        # Epsilon parameter
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, in_dim]
            edge_index: Edge tensor [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)
        
        # Aggregate messages from neighbors
        row, col = edge_index[0], edge_index[1]
        
        # Create empty tensor for aggregated features
        agg = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        
        # Sum neighbor features: agg[row[i]] += x[col[i]]
        agg.index_add_(0, row, x[col])
        
        # Apply GIN formula: (1 + eps) * x + agg
        out = (1 + self.eps) * x + agg
        
        # Apply MLP
        out = self.mlp(out)
        
        return out


class GINEncoder(nn.Module):
    """
    GIN encoder for obtaining node embeddings.
    
    Uses a stack of GIN layers to aggregate neighbor information
    and create node representations.
    
    Args:
        in_dim: Input node feature dimension
        hidden_dim: Hidden representation dimension
        num_layers: Number of GIN layers
        eps: Initial epsilon value for GIN
        train_eps: If True, epsilon is learnable
        dropout: Dropout probability between layers
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        eps: float = 0.0,
        train_eps: bool = True,
        dropout: float = 0.5
    ):
        super(GINEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # List of GIN layers
        self.convs = nn.ModuleList()
        
        # First layer: in_dim -> hidden_dim
        self.convs.append(GINConvLayer(in_dim, hidden_dim, eps, train_eps))
        
        # Remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GINConvLayer(hidden_dim, hidden_dim, eps, train_eps))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, in_dim]
            edge_index: Edge tensor [2, num_edges] (undirected graph)
        
        Returns:
            z: Node embedding matrix [num_nodes, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:  # Don't apply on last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class MLPEdgeDecoder(nn.Module):
    """
    MLP decoder for edge prediction.
    
    Takes concatenation of two node embeddings and predicts
    the probability of an edge between them.
    
    Args:
        hidden_dim: Node embedding dimension
        edge_hidden: Decoder hidden layer dimension
    """
    
    def __init__(self, hidden_dim: int = 64, edge_hidden: int = 64):
        super(MLPEdgeDecoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, 1)
        )
    
    def forward(self, z: torch.Tensor, node_pairs: torch.Tensor) -> torch.Tensor:
        """
        Prediction for node pairs.
        
        Args:
            z: Node embeddings [num_nodes, hidden_dim]
            node_pairs: Node pairs [num_pairs, 2]
        
        Returns:
            logits: Prediction logits [num_pairs]
        """
        u_idx = node_pairs[:, 0]
        v_idx = node_pairs[:, 1]
        
        # Get embeddings for pair nodes
        z_u = z[u_idx]  # [num_pairs, hidden_dim]
        z_v = z[v_idx]  # [num_pairs, hidden_dim]
        
        # Concatenate embeddings
        edge_feat = torch.cat([z_u, z_v], dim=1)  # [num_pairs, 2 * hidden_dim]
        
        # Predict through MLP
        logits = self.mlp(edge_feat).squeeze(-1)  # [num_pairs]
        
        return logits


class GINLinkPredictor(nn.Module):
    """
    Complete model for GIN-based link prediction.
    
    Combines GIN encoder for obtaining node embeddings
    and MLP decoder for edge prediction.
    
    Args:
        in_dim: Input node feature dimension
        hidden_dim: Hidden representation dimension
        num_layers: Number of GIN layers in encoder
        edge_hidden: Decoder hidden layer dimension
        eps: Initial epsilon value for GIN
        train_eps: If True, epsilon is learnable
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        edge_hidden: int = 64,
        eps: float = 0.0,
        train_eps: bool = True,
        dropout: float = 0.5
    ):
        super(GINLinkPredictor, self).__init__()
        
        self.encoder = GINEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            eps=eps,
            train_eps=train_eps,
            dropout=dropout
        )
        
        self.decoder = MLPEdgeDecoder(
            hidden_dim=hidden_dim,
            edge_hidden=edge_hidden
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_pairs: torch.Tensor
    ) -> torch.Tensor:
        """
        Model forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, in_dim]
            edge_index: Edge tensor [2, num_edges]
            node_pairs: Node pairs for prediction [num_pairs, 2]
        
        Returns:
            logits: Prediction logits [num_pairs] (without sigmoid)
        """
        # Get node embeddings
        z = self.encoder(x, edge_index)
        
        # Predict for pairs
        logits = self.decoder(z, node_pairs)
        
        return logits
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encoding only - get node embeddings.
        
        Args:
            x: Node feature matrix [num_nodes, in_dim]
            edge_index: Edge tensor [2, num_edges]
        
        Returns:
            z: Node embeddings [num_nodes, hidden_dim]
        """
        return self.encoder(x, edge_index)
    
    def decode(self, z: torch.Tensor, node_pairs: torch.Tensor) -> torch.Tensor:
        """
        Decoding only - predict for node pairs.
        
        Args:
            z: Node embeddings [num_nodes, hidden_dim]
            node_pairs: Node pairs [num_pairs, 2]
        
        Returns:
            logits: Prediction logits [num_pairs]
        """
        return self.decoder(z, node_pairs)
