import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 설정
# ---------------------------
DATA_DIR         = "./dataset_ad"
TRAIN_PATTERN    = os.path.join(DATA_DIR, "train_*.csv")
TEST_PATTERN     = os.path.join(DATA_DIR, "test_*.csv")
MODEL_SAVE_PATH  = "./pose_classifier.pth"
BATCH_SIZE       = 32
EPOCHS           = 30
LR               = 1e-3
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1) CSV 로드 & 통합
# ---------------------------
train_files = glob.glob(TRAIN_PATTERN)
test_files  = glob.glob(TEST_PATTERN)
if not train_files or not test_files:
    raise FileNotFoundError(f"No CSV files found under {DATA_DIR}")

df_train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
df_test  = pd.concat([pd.read_csv(f) for f in test_files ], ignore_index=True)

# 피처/레이블 분리
feature_cols = [c for c in df_train.columns if c != "label"]
X_train = df_train[feature_cols].values.astype(np.float32)
y_train = df_train["label"].values.astype(np.int64)
X_test  = df_test[feature_cols].values.astype(np.float32)
y_test  = df_test["label"].values.astype(np.int64)

num_classes = int(df_train["label"].nunique())
input_dim   = X_train.shape[1]

# ---------------------------
# 2) Dataset & DataLoader
# ---------------------------
class PoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = PoseDataset(X_train, y_train)
test_ds  = PoseDataset(X_test,  y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------------
# 3) 모델 정의 (작고 단순한 MLP)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_dim, 64, num_classes).to(DEVICE)

# ---------------------------
# 4) 손실함수 & 옵티마이저
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ---------------------------
# 5) 학습 & 검증 루프
# ---------------------------
best_acc = 0.0
best_state = None

for epoch in range(1, EPOCHS + 1):
    # -- Train --
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    avg_loss = running_loss / len(train_ds)

    # -- Eval --
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total

    print(f"Epoch {epoch}/{EPOCHS}  Loss: {avg_loss:.4f}  Test Acc: {acc:.4f}")
    # -- save best --
    if acc > best_acc:
        best_acc = acc
        best_state = model.state_dict().copy()

# ---------------------------
# 6) 최종 모델 저장
# ---------------------------
torch.save({
    "model_state_dict": best_state,
    "input_dim": input_dim,
    "num_classes": num_classes
}, MODEL_SAVE_PATH)

print(f"\n✅ Best model saved to {MODEL_SAVE_PATH}  (Test Acc: {best_acc:.4f})")
