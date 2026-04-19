import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# =====================
# 1. 读取数据
# =====================
with open("/root/autodl-tmp/audio_dataset/train_paired_dataset_787.json", "r") as f:
    data = json.load(f)

X = []
Y = []

for item in data:
    mf = item["music_feat"]
    sf = item["sfx_feat"]

    x = [
        mf["rms"],
        mf["centroid"],
        mf["low_ratio"],
        mf["mid_ratio"],
        sf["rms"],
        sf["centroid"],
        sf["low_ratio"],
        sf["bandwidth"],
        item["energy_ratio"]
    ]

    gain = item["label"]["gain"]
    cutoff = item["label"]["high_pass_cutoff"]

    # ===== label normalization =====
    gain = gain / 15.0                  # [-1, 1]
    cutoff = (cutoff - 100) / 150.0     # [0, 1]

    y = [gain, cutoff]

    X.append(x)
    Y.append(y)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# =====================
# 2. 特征归一化
# =====================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =====================
# 3. Dataset
# =====================
class MixDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = MixDataset(X, Y)

# 划分 train / val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# =====================
# 4. MLP模型
# =====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()

# =====================
# 5. 加权 Loss（⭐重点）
# =====================
def weighted_loss(pred, target):
    gain_loss = (pred[:, 0] - target[:, 0]) ** 2
    cutoff_loss = (pred[:, 1] - target[:, 1]) ** 2

    # gain 权重 ×2
    return (2.0 * gain_loss + 1.0 * cutoff_loss).mean()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =====================
# 6. 训练（带 Early Stopping）
# =====================
EPOCHS = 100
patience = 10

best_val_loss = float("inf")
counter = 0

for epoch in range(EPOCHS):
    # ===== 训练 =====
    model.train()
    train_loss = 0

    for x, y in train_loader:
        pred = model(x)
        loss = weighted_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)   # ⭐ 更规范（平均）

    # ===== 验证 =====
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            loss = weighted_loss(pred, y)
            val_loss += loss.item()

    val_loss /= len(val_loader)       # ⭐ 平均

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ===== Early Stopping =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0

        # ⭐ 保存最佳模型
        torch.save(model.state_dict(), "/root/autodl-tmp/models/best_model.pth")
        print("✅ Save best model")

    else:
        counter += 1
        print(f"⚠️ No improvement ({counter}/{patience})")

    if counter >= patience:
        print("🛑 Early stopping triggered")
        break
        
# =====================
# 7. 保存模型 & scaler
# =====================
model.load_state_dict(torch.load("/root/autodl-tmp/models/best_model.pth"))

torch.save(model.state_dict(), "/root/autodl-tmp/models/mix_model.pth")

# 保存 scaler（非常重要）
import pickle
with open("/root/autodl-tmp/models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ 训练完成 & 已保存模型")