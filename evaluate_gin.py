"""
Script for training and evaluating GIN model on SemanticGraph datasets.

Usage:
    python evaluate_gin.py --dataset_path SemanticGraph_delta_1_cutoff_25_minedge_1.pkl

Parameters:
    --dataset_path: path to pickle file with dataset
    --epochs: number of training epochs (default: 10)
    --hidden_dim: hidden layer dimension (default: 64)
    --num_layers: number of GIN layers (default: 3)
    --lr: learning rate (default: 1e-3)
    --batch_size: batch size (default: 2048)
    --edge_hidden: decoder hidden layer dimension (default: 64)
    --weight_decay: L2 regularization (default: 1e-5)
    --dropout: dropout probability (default: 0.5)
    --device: device for computations (default: auto)
    --seed: random seed (default: 42)
    --use_learnable_emb: use learnable embeddings instead of ones
    --emb_dim: learnable embeddings dimension (default: 16)
"""

import argparse
import os
import pickle
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from gin_model import GINLinkPredictor
from utils import NUM_OF_VERTICES, calculate_ROC


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_semantic_graph(dataset_path: str):
    """
    Load SemanticGraph dataset from pickle file.
    
    Args:
        dataset_path: path to pickle file
    
    Returns:
        full_dynamic_graph_sparse: full graph (numpy array dim(n,3))
        unconnected_vertex_pairs: vertex pairs for prediction
        unconnected_vertex_pairs_solution: labels (0/1)
        year_start: start year
        years_delta: years delta for prediction
        vertex_degree_cutoff: minimum vertex degree
        min_edges: minimum number of edges
    """
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    (full_dynamic_graph_sparse, unconnected_vertex_pairs, 
     unconnected_vertex_pairs_solution, year_start, years_delta, 
     vertex_degree_cutoff, min_edges) = data
    
    return (full_dynamic_graph_sparse, unconnected_vertex_pairs, 
            unconnected_vertex_pairs_solution, year_start, years_delta,
            vertex_degree_cutoff, min_edges)


def build_edge_index_from_graph(full_dynamic_graph_sparse: np.ndarray, 
                                 year_start: int,
                                 num_nodes: int = NUM_OF_VERTICES) -> torch.Tensor:
    """
    Build edge_index tensor from dynamic graph.
    
    Args:
        full_dynamic_graph_sparse: edge array [v1, v2, timestamp]
        year_start: year up to which edges are considered
        num_nodes: number of vertices in graph
    
    Returns:
        edge_index: tensor [2, num_edges] for undirected graph
    """
    from datetime import date
    
    day_origin = date(1990, 1, 1)
    day_cutoff = date(year_start, 12, 31)
    cutoff_days = (day_cutoff - day_origin).days
    
    # Filter edges up to year_start
    mask = full_dynamic_graph_sparse[:, 2] < cutoff_days
    edges = full_dynamic_graph_sparse[mask][:, :2]
    
    # Create edge list (undirected graph - add both directions)
    edge_list = []
    for v1, v2 in edges:
        edge_list.append([int(v1), int(v2)])
        edge_list.append([int(v2), int(v1)])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return edge_index


def prepare_data_splits(unconnected_vertex_pairs: np.ndarray,
                        unconnected_vertex_pairs_solution: np.ndarray,
                        train_ratio: float = 0.9,
                        seed: int = 42):
    """
    Split data into train/test sets with class balance preservation.
    
    Args:
        unconnected_vertex_pairs: vertex pairs array [num_pairs, 2]
        unconnected_vertex_pairs_solution: labels [num_pairs]
        train_ratio: train set ratio
        seed: random seed
    
    Returns:
        train_pairs, train_labels, test_pairs, test_labels
    """
    np.random.seed(seed)
    
    # Split by classes for stratified splitting
    pos_idx = np.where(unconnected_vertex_pairs_solution == 1)[0]
    neg_idx = np.where(unconnected_vertex_pairs_solution == 0)[0]
    
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    
    # Split each class
    pos_train_size = int(len(pos_idx) * train_ratio)
    neg_train_size = int(len(neg_idx) * train_ratio)
    
    train_idx = np.concatenate([pos_idx[:pos_train_size], neg_idx[:neg_train_size]])
    test_idx = np.concatenate([pos_idx[pos_train_size:], neg_idx[neg_train_size:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    train_pairs = unconnected_vertex_pairs[train_idx]
    train_labels = unconnected_vertex_pairs_solution[train_idx]
    test_pairs = unconnected_vertex_pairs[test_idx]
    test_labels = unconnected_vertex_pairs_solution[test_idx]
    
    return train_pairs, train_labels, test_pairs, test_labels


def create_node_features(num_nodes: int, 
                         mode: str = 'ones',
                         emb_dim: int = 16,
                         device: torch.device = None) -> tuple:
    """
    Create node features.
    
    Args:
        num_nodes: number of nodes
        mode: 'ones' or 'learnable'
        emb_dim: embedding dimension (for mode='learnable')
        device: device
    
    Returns:
        x: feature tensor or None (if learnable)
        embedding_layer: embedding layer or None
    """
    if mode == 'ones':
        x = torch.ones(num_nodes, 1, device=device)
        return x, None
    elif mode == 'learnable':
        embedding_layer = nn.Embedding(num_nodes, emb_dim).to(device)
        return None, embedding_layer
    else:
        raise ValueError(f"Unknown mode: {mode}")


def train_epoch(model: GINLinkPredictor,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                train_pairs: torch.Tensor,
                train_labels: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                batch_size: int,
                device: torch.device,
                embedding_layer: nn.Module = None) -> float:
    """
    Single training epoch.
    
    Args:
        model: GINLinkPredictor model
        x: node features
        edge_index: edge tensor
        train_pairs: training pairs
        train_labels: labels
        optimizer: optimizer
        criterion: loss function
        batch_size: batch size
        device: device
        embedding_layer: embedding layer (optional)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    # Shuffle data
    num_samples = len(train_pairs)
    perm = torch.randperm(num_samples)
    train_pairs = train_pairs[perm]
    train_labels = train_labels[perm]
    
    total_loss = 0.0
    num_batches = 0
    
    for i in range(0, num_samples, batch_size):
        batch_pairs = train_pairs[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size].float()
        
        optimizer.zero_grad()
        
        # Get features
        if embedding_layer is not None:
            node_ids = torch.arange(x.size(0) if x is not None else NUM_OF_VERTICES, device=device)
            x_input = embedding_layer(node_ids)
        else:
            x_input = x
        
        # Forward pass
        logits = model(x_input, edge_index, batch_pairs)
        
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model: GINLinkPredictor,
             x: torch.Tensor,
             edge_index: torch.Tensor,
             test_pairs: torch.Tensor,
             test_labels: torch.Tensor,
             batch_size: int,
             device: torch.device,
             embedding_layer: nn.Module = None,
             use_baseline_auc: bool = True) -> tuple:
    """
    Evaluate model on test set.
    
    Args:
        model: GINLinkPredictor model
        x: node features
        edge_index: edge tensor
        test_pairs: test pairs
        test_labels: labels
        batch_size: batch size
        device: device
        embedding_layer: embedding layer (optional)
        use_baseline_auc: use calculate_ROC from utils.py (as in baseline)
    
    Returns:
        auc: AUC-ROC score
        predictions: predicted probabilities (sorted indices)
    """
    model.eval()
    
    all_logits = []
    num_samples = len(test_pairs)
    
    for i in range(0, num_samples, batch_size):
        batch_pairs = test_pairs[i:i+batch_size]
        
        # Get features
        if embedding_layer is not None:
            node_ids = torch.arange(x.size(0) if x is not None else NUM_OF_VERTICES, device=device)
            x_input = embedding_layer(node_ids)
        else:
            x_input = x
        
        logits = model(x_input, edge_index, batch_pairs)
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)
    predictions = all_logits.cpu().numpy()
    labels = test_labels.cpu().numpy()
    
    # Create sorted index list (from highest prediction to lowest)
    sorted_indices = np.flip(np.argsort(predictions, axis=0))
    
    if use_baseline_auc:
        # Use calculate_ROC from utils.py as in baseline
        import matplotlib
        matplotlib.use('Agg')  # Disable interactive mode for plt
        auc = calculate_ROC(sorted_indices, labels)
    else:
        # Simple AUC calculation via trapezoids (if calculate_ROC unavailable)
        auc = compute_auc_simple(predictions, labels)
    
    return auc, sorted_indices


def compute_auc_simple(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Simple AUC-ROC calculation without external dependencies.
    
    Args:
        predictions: predictions array
        labels: labels array (0/1)
    
    Returns:
        AUC score
    """
    # Sort by predictions
    sorted_idx = np.argsort(-predictions)
    labels_sorted = labels[sorted_idx]
    
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Calculate AUC via Wilcoxon-Mann-Whitney statistic
    rank_sum = 0.0
    fp = 0
    
    for i, label in enumerate(labels_sorted):
        if label == 0:
            fp += 1
        else:
            rank_sum += fp
    
    auc = 1.0 - rank_sum / (n_pos * n_neg)
    return auc


def write_log(log_path: str, message: str, also_print: bool = True):
    """Write message to log file and optionally print to screen."""
    if also_print:
        print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def main():
    parser = argparse.ArgumentParser(description='GIN Link Prediction on SemanticGraph')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SemanticGraph pickle file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for GIN')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GIN layers')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size')
    parser.add_argument('--edge_hidden', type=int, default=64,
                        help='Hidden dimension for edge decoder')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_learnable_emb', action='store_true',
                        help='Use learnable embeddings instead of ones')
    parser.add_argument('--emb_dim', type=int, default=16,
                        help='Embedding dimension (if using learnable embeddings)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of data for training')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Form log file path
    dataset_name = os.path.basename(args.dataset_path).replace('.pkl', '')
    log_path = f"logs_{dataset_name}_GIN.txt"
    
    # Start logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_log(log_path, f"=" * 60)
    write_log(log_path, f"GIN Link Prediction - {timestamp}")
    write_log(log_path, f"=" * 60)
    write_log(log_path, f"Dataset: {args.dataset_path}")
    write_log(log_path, f"Parameters:")
    write_log(log_path, f"  epochs: {args.epochs}")
    write_log(log_path, f"  hidden_dim: {args.hidden_dim}")
    write_log(log_path, f"  num_layers: {args.num_layers}")
    write_log(log_path, f"  lr: {args.lr}")
    write_log(log_path, f"  batch_size: {args.batch_size}")
    write_log(log_path, f"  edge_hidden: {args.edge_hidden}")
    write_log(log_path, f"  weight_decay: {args.weight_decay}")
    write_log(log_path, f"  dropout: {args.dropout}")
    write_log(log_path, f"  seed: {args.seed}")
    write_log(log_path, f"  use_learnable_emb: {args.use_learnable_emb}")
    if args.use_learnable_emb:
        write_log(log_path, f"  emb_dim: {args.emb_dim}")
    write_log(log_path, f"  train_ratio: {args.train_ratio}")
    write_log(log_path, f"  device: {device}")
    write_log(log_path, "")
    
    # Load data
    write_log(log_path, "Loading dataset...")
    start_time = time.time()
    
    (full_dynamic_graph_sparse, unconnected_vertex_pairs,
     unconnected_vertex_pairs_solution, year_start, years_delta,
     vertex_degree_cutoff, min_edges) = load_semantic_graph(args.dataset_path)
    
    write_log(log_path, f"  year_start: {year_start}")
    write_log(log_path, f"  years_delta: {years_delta}")
    write_log(log_path, f"  vertex_degree_cutoff: {vertex_degree_cutoff}")
    write_log(log_path, f"  min_edges: {min_edges}")
    write_log(log_path, f"  num_vertex_pairs: {len(unconnected_vertex_pairs)}")
    write_log(log_path, f"  positive_pairs: {sum(unconnected_vertex_pairs_solution)}")
    write_log(log_path, f"  negative_pairs: {len(unconnected_vertex_pairs_solution) - sum(unconnected_vertex_pairs_solution)}")
    
    # Build edge_index
    write_log(log_path, "\nBuilding edge_index...")
    edge_index = build_edge_index_from_graph(
        full_dynamic_graph_sparse, year_start, NUM_OF_VERTICES
    )
    write_log(log_path, f"  num_edges (directed): {edge_index.size(1)}")
    
    # Prepare train/test splits
    write_log(log_path, "\nPreparing train/test splits...")
    train_pairs, train_labels, test_pairs, test_labels = prepare_data_splits(
        unconnected_vertex_pairs, unconnected_vertex_pairs_solution,
        args.train_ratio, args.seed
    )
    
    write_log(log_path, f"  train_pairs: {len(train_pairs)}")
    write_log(log_path, f"  train_positive: {sum(train_labels)}")
    write_log(log_path, f"  test_pairs: {len(test_pairs)}")
    write_log(log_path, f"  test_positive: {sum(test_labels)}")
    
    # Convert to tensors
    train_pairs = torch.tensor(train_pairs, dtype=torch.long, device=device)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=device)
    test_pairs = torch.tensor(test_pairs, dtype=torch.long, device=device)
    test_labels = torch.tensor(test_labels, dtype=torch.long, device=device)
    edge_index = edge_index.to(device)
    
    # Create node features
    write_log(log_path, "\nCreating node features...")
    feature_mode = 'learnable' if args.use_learnable_emb else 'ones'
    x, embedding_layer = create_node_features(
        NUM_OF_VERTICES, mode=feature_mode, 
        emb_dim=args.emb_dim, device=device
    )
    
    in_dim = args.emb_dim if args.use_learnable_emb else 1
    write_log(log_path, f"  feature_mode: {feature_mode}")
    write_log(log_path, f"  in_dim: {in_dim}")
    
    load_time = time.time() - start_time
    write_log(log_path, f"\nData loading completed in {load_time:.2f}s")
    
    # Create model
    write_log(log_path, "\nInitializing model...")
    model = GINLinkPredictor(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_hidden=args.edge_hidden,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if embedding_layer is not None:
        num_params += sum(p.numel() for p in embedding_layer.parameters())
    write_log(log_path, f"  total_parameters: {num_params:,}")
    
    # Optimizer and loss
    params = list(model.parameters())
    if embedding_layer is not None:
        params += list(embedding_layer.parameters())
    
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training
    write_log(log_path, "\n" + "=" * 40)
    write_log(log_path, "Starting training...")
    write_log(log_path, "=" * 40)
    
    best_auc = 0.0
    best_epoch = 0
    epoch_results = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_loss = train_epoch(
            model, x, edge_index, train_pairs, train_labels,
            optimizer, criterion, args.batch_size, device, embedding_layer
        )
        
        # Evaluation
        train_auc, _ = evaluate(
            model, x, edge_index, train_pairs, train_labels,
            args.batch_size, device, embedding_layer
        )
        test_auc, _ = evaluate(
            model, x, edge_index, test_pairs, test_labels,
            args.batch_size, device, embedding_layer
        )
        
        epoch_time = time.time() - epoch_start
        
        epoch_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'test_auc': test_auc
        })
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch = epoch
        
        write_log(log_path, 
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train AUC: {train_auc:.4f} | "
            f"Test AUC: {test_auc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
    
    # Final results
    write_log(log_path, "\n" + "=" * 40)
    write_log(log_path, "Training completed!")
    write_log(log_path, "=" * 40)
    write_log(log_path, f"Best Test AUC: {best_auc:.4f} (epoch {best_epoch})")
    
    # Final evaluation on all data
    final_auc, final_predictions = evaluate(
        model, x, edge_index, test_pairs, test_labels,
        args.batch_size, device, embedding_layer
    )
    
    write_log(log_path, f"\nFinal Test AUC: {final_auc:.4f}")
    write_log(log_path, f"Log saved to: {log_path}")
    
    # Save results in baseline-compatible format
    write_log(log_path, "\n" + "-" * 40)
    write_log(log_path, "Summary (baseline-compatible format):")
    write_log(log_path, f"delta={years_delta}")
    write_log(log_path, f"cutoff={vertex_degree_cutoff}")
    write_log(log_path, f"min_edges={min_edges}")
    write_log(log_path, f"hidden_dim={args.hidden_dim}")
    write_log(log_path, f"num_layers={args.num_layers}")
    write_log(log_path, f"epochs={args.epochs}")
    write_log(log_path, f"AUC={final_auc:.6f}")
    
    return final_auc


if __name__ == '__main__':
    main()
