import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import AlphabetDataset

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "dataset_augmented"
MODEL_DIR = "models"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
NUM_CLASSES = 26
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Device:", DEVICE)
print("Saving models to:", os.path.abspath(MODEL_DIR))

# -----------------------------
# DATA
# -----------------------------
dataset = AlphabetDataset(DATASET_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -----------------------------
# MODEL (TCN)
# -----------------------------
class AlphabetTCN(nn.Module):
    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)

model = AlphabetTCN(num_classes=NUM_CLASSES).to(DEVICE)

# -----------------------------
# TRAINING SETUP
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0
epochs_no_improve = 0

BEST_PATH = os.path.join(MODEL_DIR, "alphabet_tcn_best.pth")
LAST_PATH = os.path.join(MODEL_DIR, "alphabet_tcn_last.pth")

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_acc = correct / total if total else 0.0

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Acc: {val_acc:.2%}"
    )

    # Save last checkpoint
    torch.save(model.state_dict(), LAST_PATH)

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
        print(f"✅ New best model saved ({best_val_acc:.2%})")
    else:
        epochs_no_improve += 1
        print(f"⚠ No improvement ({epochs_no_improve}/{PATIENCE})")

    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print("⛔ Early stopping triggered")
        break

print("\nTraining complete.")
print(f"Best model: {os.path.abspath(BEST_PATH)} | Best Val Acc: {best_val_acc:.2%}")
print(f"Last model: {os.path.abspath(LAST_PATH)}")
