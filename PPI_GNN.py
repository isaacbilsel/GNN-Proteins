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
from torch_geometric.nn import SAGEConv, GraphNorm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, GATv2Conv
import copy

# Load train, val, and test sets
train_dataset = PPI(root='data/PPI', split='train')
val_dataset = PPI(root='data/PPI', split='val')
test_dataset = PPI(root='data/PPI', split='test')

# Wrap in data loaders for batch training
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of nodes in first graph: {train_dataset[0].num_nodes}")
print(f"Feature shape: {train_dataset[0].x.shape}")
print(f"Labels shape: {train_dataset[0].y.shape}")

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
# plt.show()

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

class HybridGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_classes, heads=4):
        super(HybridGNN, self).__init__()
        self.sage_branch = DeepGraphSAGE(in_feats, hidden_feats, out_feats)
        self.gat_branch = GAT(in_feats, hidden_feats, out_feats, heads=heads)

        # Final classifier after concatenating both GNN outputs
        self.classifier = nn.Linear(2 * out_feats, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # Get DeepGraphSAGE output
        out_sage = self.sage_branch(x, edge_index, batch)

        # Get GAT output
        out_gat = self.gat_branch(x, edge_index)

        # Concatenate along feature dimension
        out = torch.cat([out_sage, out_gat], dim=1)

        # Optional classifier and sigmoid for multi-label classification
        out = self.dropout(out)
        out = self.classifier(out)
        # out = torch.sigmoid(out)
        return out

# Evaluation helper
def evaluate(model, loader):
    model.eval()
    total_f1 = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = torch.sigmoid(model(batch.x, batch.edge_index))  # Apply sigmoid here for weigted BCE loss. # batch argument for graphnorm
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
            out = model(batch.x, batch.edge_index)     # batch argument for graphnorm
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

#### Driver Code ####
"""
## Run Hybrid GAT-GraphSage model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_feats = train_dataset.num_node_features
out_feats = train_dataset.num_classes
hidden_feats = 256
model = HybridGNN(in_feats, hidden_feats, out_feats, 121).to(device)
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

"""
# Run GAT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_feats = train_dataset.num_node_features
out_feats = train_dataset.num_classes
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

# Run GraphSAGE 
# Init model, optimizer, loss
in_feats = train_dataset.num_node_features
out_feats = train_dataset.num_classes
hidden_feats = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(in_feats, hidden_feats, out_feats).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)                # Changed from:  loss_fn = nn.BCELoss()

# Train
train_losses, val_f1_scores = train_model(model, train_loader, val_loader, optimizer, loss_fn)

# Final test evaluation
val_f1 = evaluate(model, val_loader)
test_f1 = evaluate(model, test_loader)
print(f"\nValidation F1: {val_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
# plot_graphs(train_losses, val_f1_scores)

"""
# Run DeepGraphSAGE
# Init
in_feats = train_dataset.num_node_features
out_feats = train_dataset.num_classes
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
