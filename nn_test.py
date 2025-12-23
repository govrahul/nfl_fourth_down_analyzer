import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import modeling_nn
import data_load
import warnings
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Load saved nn
model = modeling_nn.FourthDownNN(input_size=5)
model.load_state_dict(torch.load('Models/fourth_down_nn_model.pth'))
model.eval()

# Load data
fourth_downs = data_load.load_fourth_down_data('Data/play_by_play_*.csv')
X = fourth_downs[['ydstogo', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining', 'yardline_100']]
y = fourth_downs['go']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Make predictions on test set
test_loader = DataLoader(TensorDataset(torch.tensor(X_test.values, dtype=torch.float32),
                                        torch.tensor(y_test.values, dtype=torch.long)),
                            batch_size=32)

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

# calculate go percentage from predictions and data
print("Predicted go rate:", preds.mean())
print("True go rate:", y_test.mean())