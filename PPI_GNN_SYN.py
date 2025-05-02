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
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
import copy

# Load train, val, and test sets
train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')

import networkx as nx
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

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
num_synthetic_graphs = 5
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
    total_pos = torch.zeros(out_feats)
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

# GraphSAGE model definition
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.conv2 = SAGEConv(hidden_feats, out_feats)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
# DeepGraphSAGE model definition
class DeepGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(DeepGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats)
        self.bn1 = nn.BatchNorm1d(hidden_feats)

        self.conv2 = SAGEConv(hidden_feats, hidden_feats)
        self.bn2 = nn.BatchNorm1d(hidden_feats)

        self.conv3 = SAGEConv(hidden_feats, hidden_feats)
        self.bn3 = nn.BatchNorm1d(hidden_feats)

        self.conv4 = SAGEConv(hidden_feats, out_feats)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

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

        x = self.conv4(x, edge_index)  # NOTE: no sigmoid here
        return x
    
# GAT model definition
class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads=heads)
        self.gat2 = GATConv(hidden_feats * heads, hidden_feats, heads=heads)
        self.gat3 = GATConv(hidden_feats * heads, out_feats, heads=1)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index)
        return x

# Evaluation helper
def evaluate(model, loader):
    model.eval()
    total_f1 = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = torch.sigmoid(model(batch.x, batch.edge_index))  # Apply sigmoid here
            preds = (out > 0.5).float()
            f1 = f1_score(batch.y.cpu().numpy(), preds.cpu().numpy(), average='micro')
            total_f1 += f1
    return total_f1 / len(loader)

# Training function
def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=100, patience=15):
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
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1 = evaluate(model, val_loader)

        train_losses.append(avg_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")

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

#### Driver Code ####
# Run GAT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_feats = combined_train_dataset[0].x.shape[1]
out_feats = combined_train_dataset[0].y.shape[1]
hidden_feats = 512
model = GAT(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))       # Changed from:  loss_fn = nn.BCELoss()

# Train the GAT model
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=100, patience=20)

# Final test performance
val_f1 = evaluate(model, val_loader)
test_f1 = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
plot_graphs(train_losses, val_f1_scores)

"""
# Run GraphSAGE 
# Init model, optimizer, loss
in_feats = combined_train_dataset.num_node_features
out_feats = combined_train_dataset.num_classes
hidden_feats = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)                # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Final test evaluation
val_f1 = evaluate(model, val_loader)
test_f1 = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
plot_graphs(train_losses, val_f1_scores)
"""

"""
# Run DeepGraphSAGE
# Init
in_feats = combined_train_dataset.num_node_features
out_feats = combined_train_dataset.num_classes
hidden_feats = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepGraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))       # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Validate and Test
val_f1 = evaluate(model, val_loader)
test_f1 = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
plot_graphs(train_losses, val_f1_scores)

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
