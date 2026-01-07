# =========================
# OFFLINE SAFETY
# =========================
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = "/root/.cache/torch"

# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DINOv2
# =========================
from transformers import AutoModel

DINO_PATH = "/kaggle/input/dinov2/pytorch/small/1"
dinov2 = AutoModel.from_pretrained(DINO_PATH, local_files_only=True).to(device)

# 🔓 Unfreeze last 2 blocks
for block in dinov2.encoder.layer[-2:]:
    for p in block.parameters():
        p.requires_grad = True

# =========================
# DATA PREP
# =========================
def prepare_data(df):
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    # 🔥 LOG TARGET
    df["target"] = np.log1p(df["target"])

    le_state = LabelEncoder()
    le_species = LabelEncoder()
    df["State_encoded"] = le_state.fit_transform(df["State"])
    df["Species_encoded"] = le_species.fit_transform(df["Species"])
    return df, le_state, le_species

# =========================
# DATASET
# =========================
class GrassDataset(Dataset):
    def __init__(self, df, image_base_dir, transform, mean, std, train=True):
        self.df = df
        self.base = Path(image_base_dir)
        self.transform = transform
        self.train = train
        self.mean = mean
        self.std = std

        self.targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
        self.groups = df.groupby("image_id")
        self.ids = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        g = self.groups.get_group(img_id)

        img = Image.open(self.base / g.iloc[0]["image_path"]).convert("RGB")
        img = self.transform(img)

        meta = torch.tensor([
            g.iloc[0]["Pre_GSHH_NDVI"],
            g.iloc[0]["Height_Ave_cm"],
            g.iloc[0]["State_encoded"],
            g.iloc[0]["Species_encoded"]
        ], dtype=torch.float32)

        if not self.train:
            return img, meta, img_id

        y = []
        for i, t in enumerate(self.targets):
            val = g[g["target_name"] == t]["target"].values[0]
            y.append((val - self.mean[i]) / self.std[i])

        return img, meta, torch.tensor(y, dtype=torch.float32)

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# MODEL
# =========================
class BiomassPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = dinov2
        self.head = nn.Sequential(
            nn.Linear(384+4, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)
        )

    def forward(self, x, meta):
        feat = self.backbone(pixel_values=x).last_hidden_state[:,0]
        return self.head(torch.cat([feat, meta], 1))

# =========================
# TRAIN
# =========================
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for x, m, y in tqdm(loader):
        x,m,y = x.to(device), m.to(device), y.to(device)
        p = model(x,m)
        loss = loss_fn(p,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv("/kaggle/input/csiro-biomass/train.csv")
    df, le_state, le_species = prepare_data(df)

    targets = ["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"]
    stats = df.groupby("target_name")["target"].agg(["mean","std"]).loc[targets]
    mean, std = stats["mean"].values, stats["std"].values + 1e-6

    ds = GrassDataset(df,"/kaggle/input/csiro-biomass",transform,mean,std)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)

    model = BiomassPredictor().to(device)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 25)
    loss_fn = nn.SmoothL1Loss()

    for e in range(25):
        l = train_epoch(model, dl, opt, loss_fn)
        sched.step()
        print(f"Epoch {e+1}: {l:.4f}")

    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std,
        "state": le_state,
        "species": le_species
    }, "model.pth")

    return model, mean, std

# =========================
# PREDICT
# =========================
def predict(model, mean, std):
    df = pd.read_csv("/kaggle/input/csiro-biomass/test.csv")
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df["State_encoded"]=0; df["Species_encoded"]=0
    df["Pre_GSHH_NDVI"]=0.5; df["Height_Ave_cm"]=10.0

    ds = GrassDataset(df,"/kaggle/input/csiro-biomass",transform,mean,std,train=False)
    model.eval()
    out=[]
    with torch.no_grad():
        for x,m,i in tqdm(ds):
            p = model(x.unsqueeze(0).to(device), m.unsqueeze(0).to(device))[0]
            p = np.expm1(p.cpu().numpy()*std + mean)
            for t,v in zip(["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"],p):
                out.append({"sample_id":f"{i}__{t}","target":float(v)})

    pd.DataFrame(out).to_csv("submission.csv", index=False)

# =========================
# RUN
# =========================
model, mean, std = main()
predict(model, mean, std)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# =========================
# IMAGE PATH
# =========================
img_path = "/kaggle/input/csiro-biomass/test/ID1001187975.jpg"
img = Image.open(img_path).convert("RGB")

# =========================
# TRANSFORM (same as training)
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
img_tensor = transform(img).unsqueeze(0).to(device)

# =========================
# DUMMY METADATA
# =========================
meta = torch.tensor([[0.5, 10.0, 0, 0]], dtype=torch.float32).to(device)

# =========================
# PREDICTION
# =========================
model.eval()
with torch.no_grad():
    pred = model(img_tensor, meta)[0]
    pred = np.expm1(pred.cpu().numpy() * std + mean)

# =========================
# VISUALIZATION
# =========================
targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

fig, axes = plt.subplots(1, 2, figsize=(14,6))

# --- Left: Image ---
axes[0].imshow(img)
axes[0].axis('off')
axes[0].set_title("Test Image", fontsize=14)

# --- Right: Bar chart ---
bars = axes[1].bar(targets, pred, color=['green','brown','yellow','blue','purple'])
axes[1].set_title("Predicted Biomass", fontsize=14)
axes[1].set_ylabel("Biomass (g)")
axes[1].set_xlabel("Target")
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()
