import os
import random
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

from PIL import Image  
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from models.model import build_resnet18, save_checkpoint, load_checkpoint
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cpu")
print("Using device:", device)

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="../../data/train")
parser.add_argument("--val_dir",   type=str, default="../../data/val")
parser.add_argument("--test_dir",  type=str, default="../../data/test")
parser.add_argument("--img_size",  type=int, default=64)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs",     type=int, default=50)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--out_model",  type=str, default="best_resnet18_rgb.pth")
parser.add_argument("--out_report", type=str, default="test_results.txt")
parser.add_argument(
    "--keep_classes",
    type=str,
    default='beach,buildings,forest,harbor,freeway',
    
)

args = parser.parse_args()

set_seed(args.seed)
IMG_SIZE = args.img_size

tf_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
   
    transforms.ToTensor(),
])

tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()                 
])


def subset_imagefolder(dataset, keep_class_names):


    orig_classes = list(dataset.classes)

 
    new_classes = list(keep_class_names)
    new_class_to_idx = {c: i for i, c in enumerate(new_classes)}

    new_samples = []
    new_targets = []

    for path, old_label in dataset.samples:
        cls_name = orig_classes[old_label]
        if cls_name in new_class_to_idx:
            new_label = new_class_to_idx[cls_name]
            new_samples.append((path, new_label))
            new_targets.append(new_label)

    dataset.samples = new_samples
    dataset.targets = new_targets
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx

    return dataset
train_ds = datasets.ImageFolder(args.train_dir, transform=tf_train)
val_ds   = datasets.ImageFolder(args.val_dir,   transform=tf_eval)
test_ds  = datasets.ImageFolder(args.test_dir,  transform=tf_eval)


keep_cls_list = [c.strip() for c in args.keep_classes.split(",") if c.strip()]


train_ds = subset_imagefolder(train_ds, keep_cls_list)
val_ds   = subset_imagefolder(val_ds,   keep_cls_list)
test_ds  = subset_imagefolder(test_ds,  keep_cls_list)


num_classes = len(train_ds.classes)

train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
test_loader = DataLoader(
    test_ds,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = build_resnet18(
    num_classes=num_classes,
    pretrained=True,
    dropout=0.0,
    freeze_backbone=True,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

EPOCHS = args.epochs
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    loss_sum = 0.0
    y_true_train, y_pred_train = [], []

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        y_pred_train += logits.argmax(1).detach().cpu().tolist()
        y_true_train += labels.detach().cpu().tolist()

    train_loss = loss_sum / len(train_loader)
    train_acc = accuracy_score(y_true_train, y_pred_train)
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            preds = logits.argmax(1)

            val_pred += preds.cpu().tolist()
            val_true += labels.cpu().tolist()

    val_acc = accuracy_score(val_true, val_pred)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, args.out_model)
        print(f"  -> New best model saved to: {args.out_model} (val_acc={best_val_acc:.4f})")
best_model = build_resnet18(
    num_classes=num_classes,
    pretrained=False,     
    dropout=0.0,
    freeze_backbone=False
).to(device)

load_checkpoint(best_model, args.out_model, map_location=device)
best_model.eval()

y_true, y_pred = [], []
y_prob_rows = []  # [N, C]

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device, non_blocking=True)
        logits = best_model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().tolist()

        y_true += labels.tolist()
        y_pred += preds
        y_prob_rows += probs.tolist()

y_prob = np.array(y_prob_rows)

acc = accuracy_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred, average='macro')
auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

print(f"\nTest Results:")
print(f"  Acc      = {acc:.4f}")
print(f"  F1(macro)= {f1:.4f}")
print(f"  AUC(OVR) = {auc:.4f}")
with open(args.out_report, "w") as f:
    f.write("Test results (best model based on val_acc)\n")
    f.write(f"Accuracy      : {acc:.6f}\n")
    f.write(f"F1 (macro)    : {f1:.6f}\n")
    f.write(f"AUC (OVR)     : {auc:.6f}\n")
    f.write(f"Best val_acc  : {best_val_acc:.6f}\n")
    f.write(f"Num classes   : {num_classes}\n")
    f.write(f"Used classes  : {train_ds.classes}\n")
    f.write(f"Train dir     : {args.train_dir}\n")
    f.write(f"Val dir       : {args.val_dir}\n")
    f.write(f"Test dir      : {args.test_dir}\n")

print(f"\nTest results saved to: {args.out_report}")
print(f"Best model saved to:   {args.out_model}")