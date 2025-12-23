import pandas as pd
import torch
import torch.nn as nn
import data_load
from sklearn.model_selection import train_test_split
import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define the neural network model
class FourthDownNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

# Load and split data   
def load_data():
    combined = data_load.load_fourth_down_data(path_pattern='Data/play_by_play_*.csv', download=False)
    X = combined[['ydstogo', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining', 'yardline_100']]
    y = combined['go']
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Train model with early stopping
def train_model(model, train_loader, val_loader=None, epochs=10, lr=0.001, device='cpu', patience=5):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_train_loss = epoch_loss / total
        train_acc = correct / total

        # --- Validation ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    outputs = model(X_val)
                    loss = criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == y_val).sum().item()
                    val_total += y_val.size(0)

            avg_val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

            # --- Early stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Optionally save best model state
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Validation loss did not improve for {patience} epochs. Stopping early.")
                    model.load_state_dict(best_model_state)
                    return model

        else:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")

    # If no early stopping triggered, return the last model
    if val_loader is not None:
        model.load_state_dict(best_model_state)
    return model

# Driver function
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    input_size = X_train.shape[1]

    model = FourthDownNN(input_size=input_size)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train.values, dtype=torch.float32),
                                        torch.tensor(y_train.values, dtype=torch.long)),
                          batch_size=32, shuffle=True)

    val_loader = DataLoader(TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                                        torch.tensor(y_val.values, dtype=torch.long)),
                            batch_size=32)

    test_loader = DataLoader(TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                        torch.tensor(y_test.values, dtype=torch.long)),
                            batch_size=32)
    
    model = train_model(model, train_loader, val_loader, epochs=50)

    # Evaluate on test data
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to('cpu')
            y_batch = y_batch.to('cpu')

            outputs = model(X_batch)

            # Convert logits -> probabilities
            probs = F.softmax(outputs, dim=1)[:, 1]  # P(go=1)

            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())

    # Compute metrics
    test_auc = roc_auc_score(all_labels, all_probs)

    # accuracy at 0.5 threshold
    preds = (torch.tensor(all_probs) >= 0.5).int().numpy()
    test_acc = (preds == all_labels).mean()

    print(f"Test AUC={test_auc:.4f}, Test Acc={test_acc:.4f}")

    # save model
    os.makedirs('Models', exist_ok=True)
    torch.save(model.state_dict(), 'Models/fourth_down_nn_model.pth')
