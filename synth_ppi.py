# Install all PyTorch Geometric dependencies (CPU version)
"""
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install scikit-learn
pip install matplotlib
"""


# Import libraries and load data
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GraphNorm, GCNConv, GATConv, MessagePassing
from torch_geometric.utils import softmax, scatter
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import copy

import networkx as nx
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

# Load train, val, and test sets
train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')

# ---- SYNTHETIC DATA GENERATION ---- #
# Compute feature statistics from real data
feat_mean = train_dataset[0].x.mean(dim=0)
feat_std = train_dataset[0].x.std(dim=0)

# Synthetic graph generator
def generate_synthetic_pyg_graph(num_nodes, in_feats, num_classes, avg_degree=5):
    nx_graph = nx.barabasi_albert_graph(num_nodes, max(1, avg_degree // 2))
    edge_index = pyg_utils.from_networkx(nx_graph).edge_index

    x = torch.stack([
        feat_mean + torch.normal(0, 0.5, size=(in_feats,)) * feat_std
        for _ in range(num_nodes)
    ])

    y = torch.zeros(num_nodes, num_classes)
    for i in range(num_nodes):
        label_indices = torch.randperm(num_classes)[:torch.randint(1, 4, (1,)).item()]
        y[i, label_indices] = 1.0
    y = (y + (torch.rand_like(y) < 0.05)).clamp(0, 1)

    return Data(x=x, edge_index=edge_index, y=y)

# Generate synthetic graphs
num_synthetic_graphs = 20
num_nodes_per_graph = 600
synthetic_graphs = [
    generate_synthetic_pyg_graph(num_nodes_per_graph, train_dataset.num_node_features, train_dataset.num_classes)
    for _ in range(num_synthetic_graphs)
]

# Combine real + synthetic training data
combined_train_dataset = list(train_dataset) + synthetic_graphs

in_feats = combined_train_dataset[0].x.shape[1]
out_feats = combined_train_dataset[0].y.shape[1]

# Wrap in data loaders for batch training
train_loader = DataLoader(combined_train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

print(f"Number of training graphs: {len(combined_train_dataset)}")
print(f"Number of nodes in first graph: {combined_train_dataset[0].num_nodes}")
print(f"Feature shape: {combined_train_dataset[0].x.shape}")
print(f"Labels shape: {combined_train_dataset[0].y.shape}")

#Compute and graph class-wise positive label counts and weights
def compute_pos_weights(loader):
    total_pos = torch.zeros(train_dataset.num_classes)
    total_count = 0
    for batch in loader:
        total_pos += batch.y.sum(dim=0)
        total_count += batch.y.size(0)
    neg = total_count - total_pos
    pos_weight = neg / (total_pos + 1e-6)
    return total_pos, pos_weight

label_distribution, class_weights = compute_pos_weights(train_loader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# Plot label distribution and class weights
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].bar(range(len(label_distribution)), label_distribution.numpy())
axs[0].set_title("Label Distribution in Training Set")
axs[0].set_xlabel("Class Index")
axs[0].set_ylabel("Count")

axs[1].bar(range(len(class_weights)), class_weights.cpu().numpy())
axs[1].set_title("Class Weights for Loss Function")
axs[1].set_xlabel("Class Index")
axs[1].set_ylabel("Weight")
plt.show()

# Training curve function
def plot_graphs(train_losses, val_f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1_scores, label='Validation F1', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

#GCN Model definition
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.conv3 = GCNConv(hidden_feats, out_feats)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x

# GraphSAGE model definition
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, out_feats)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
# DeepGraphSAGE model definition -- with GraphNorm layer norm
class DeepGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(DeepGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.norm1 = GraphNorm(hidden_feats)

        self.conv2 = SAGEConv(hidden_feats, hidden_feats)
        self.norm2 = GraphNorm(hidden_feats)

        self.conv3 = SAGEConv(hidden_feats, hidden_feats)
        self.norm3 = GraphNorm(hidden_feats)

        self.conv4 = SAGEConv(hidden_feats, hidden_feats)
        self.norm4 = GraphNorm(hidden_feats)

        self.conv5 = SAGEConv(hidden_feats, out_feats)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.norm4(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index)  # NOTE: no sigmoid here
        return x

""" # DeepGraphSAGE model definition -- batchnorm
class DeepGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(DeepGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.bn1 = nn.BatchNorm1d(hidden_feats)

        self.conv2 = SAGEConv(hidden_feats, hidden_feats)
        self.bn2 = nn.BatchNorm1d(hidden_feats)

        self.conv3 = SAGEConv(hidden_feats, hidden_feats)
        self.bn3 = nn.BatchNorm1d(hidden_feats)

        self.conv4 = SAGEConv(hidden_feats, hidden_feats)
        self.bn4 = nn.BatchNorm1d(hidden_feats)

        self.conv5 = SAGEConv(hidden_feats, out_feats)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.20)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv5(x, edge_index)  # NOTE: no sigmoid here
        return x
"""

# GAT model definition
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads=heads)
        self.norm1 = GraphNorm(hidden_feats * heads)
        self.gat2 = GATConv(hidden_feats * heads, hidden_feats, heads=heads)
        self.norm2 = GraphNorm(hidden_feats * heads)
        self.gat3 = GATConv(hidden_feats * heads, hidden_feats, heads=heads)
        self.norm3 = GraphNorm(hidden_feats * heads)
        self.gat4 = GATConv(hidden_feats * heads, out_feats, heads=1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = self.norm1(x, batch)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.norm2(x, batch)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index)
        x = self.norm3(x, batch)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gat4(x, edge_index)
        return x

# Attention-based GraphSAGE layer definition
class LearnableAggregatorConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.05):
        super(LearnableAggregatorConv, self).__init__(aggr=None)  # Custom aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(out_channels, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 3)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index, ptr, size_i):
        # --- Attention aggregation ---
        x_cat = torch.cat([x_i, x_j], dim=-1)
        att_weight = self.att_mlp(x_cat).squeeze(-1)
        att_weight = softmax(att_weight, index, ptr, size_i)
        att_weight = self.dropout(att_weight)
        att_msg = att_weight.unsqueeze(-1) * x_j
        att_out = scatter(att_msg, index, dim=0, dim_size=size_i, reduce='sum')

        # --- Mean aggregation ---
        mean_out = scatter(x_j, index, dim=0, dim_size=size_i, reduce='mean')

        # --- Max aggregation ---
        max_out = scatter(x_j, index, dim=0, dim_size=size_i, reduce='max')

        # --- Combine with gating ---
        cat_aggr = torch.cat([mean_out, max_out, att_out], dim=-1)  # [N, 3D]
        gate_logits = self.gate(cat_aggr)  # [N, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [N, 3]

        # Apply weights
        fused = (
            gate_weights[:, 0:1] * mean_out +
            gate_weights[:, 1:2] * max_out +
            gate_weights[:, 2:3] * att_out
        )
        return fused

    def update(self, aggr_out, x):
        return aggr_out + x  # Optional residual
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # This should never be called, as aggregation is done manually in `message`
        return inputs

# Hybrid Attention-based GraphSage GNN Model
class HybridGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0.05):
        super(HybridGNN, self).__init__()
        self.conv1 = LearnableAggregatorConv(in_feats, hidden_feats, dropout)
        self.norm1 = GraphNorm(hidden_feats)

        self.conv2 = LearnableAggregatorConv(hidden_feats, hidden_feats, dropout)
        self.norm2 = GraphNorm(hidden_feats)

        self.conv3 = LearnableAggregatorConv(hidden_feats, hidden_feats, dropout)
        self.norm3 = GraphNorm(hidden_feats)

        self.conv4 = LearnableAggregatorConv(hidden_feats, out_feats, dropout)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x, batch)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        return x 

# Evaluation helper
def evaluate(model, loader):
    model.eval()
    total_f1 = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = torch.sigmoid(model(batch.x, batch.edge_index, batch.batch))  # Apply sigmoid here for weigted BCE loss. # batch argument for graphnorm
            preds = (out > 0.5).float()

            y_true = batch.y.cpu().numpy()
            y_pred = preds.cpu().numpy()

            f1 = f1_score(y_true, y_pred, average='micro')
            precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

            total_f1 += f1
            total_precision += precision
            total_recall += recall

    n_batches = len(loader)
    return total_f1 / n_batches, total_precision / n_batches, total_recall / n_batches

# Training function
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=125, patience=15):
    train_losses = []
    val_f1_scores = []
    best_val_f1 = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)     # batch argument for graphnorm
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1, precision, recall = evaluate(model, val_loader)

        train_losses.append(avg_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_f1_scores

# ---- Driver Code ---- #
"""# Run GCN (3-layer)
# Init model, optimizer, loss
hidden_feats = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)                # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Final test evaluation
val_f1, val_recall, val_precision = evaluate(model, val_loader)
test_f1, test_recall, test_precision = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
# plot_graphs(train_losses, val_f1_scores)
"""

# Run Hybrid Model
# Init
# in_feats = train_dataset.num_node_features
# out_feats = train_dataset.num_classes
hidden_feats = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridGNN(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))       # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Validate and Test
val_f1, val_recall, val_precision = evaluate(model, val_loader)
test_f1, test_recall, test_precision = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
plot_graphs(train_losses, val_f1_scores)

"""
# Run GAT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# in_feats = train_dataset.num_node_features
# out_feats = train_dataset.num_classes
hidden_feats = 512
model = GAT(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))       # Changed from:  loss_fn = nn.BCELoss()

# Train the GAT model
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Final test performance
val_f1, val_recall, val_precision = evaluate(model, val_loader)
test_f1, test_recall, test_precision = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
plot_graphs(train_losses, val_f1_scores)
"""

""" # Run GraphSAGE 
# Init model, optimizer, loss
hidden_feats = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)                # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Final test evaluation
val_f1, val_recall, val_precision = evaluate(model, val_loader)
test_f1, test_recall, test_precision = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
# plot_graphs(train_losses, val_f1_scores)
"""

""" # Run DeepGraphSAGE
# Init
# in_feats = train_dataset.num_node_features
# out_feats = train_dataset.num_classes
hidden_feats = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepGraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))       # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Validate and Test
val_f1, val_recall, val_precision = evaluate(model, val_loader)
test_f1, test_recall, test_precision = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
plot_graphs(train_losses, val_f1_scores)
"""

"""
# Visualize predictions for the first graph in test set
model.eval()
sample = test_dataset[0].to(device)

with torch.no_grad():
    pred = model(sample.x, sample.edge_index)
    pred_bin = (pred > 0.5).float().cpu()

# Plot true vs predicted labels for first 5 nodes
num_nodes_to_plot = 5
for i in range(num_nodes_to_plot):
    plt.figure(figsize=(10, 2))
    plt.bar(range(out_feats), sample.y[i].cpu(), alpha=0.6, label='True')
    plt.bar(range(out_feats), pred_bin[i], alpha=0.4, label='Predicted')
    plt.title(f"Node {i} - True vs Predicted Labels")
    plt.xlabel("Label Index")
    plt.ylabel("Label Presence (0 or 1)")
    plt.legend()
    plt.show()

    """