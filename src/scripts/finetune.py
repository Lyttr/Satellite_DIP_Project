import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from models.model import build_resnet18, load_checkpoint, save_checkpoint


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

 
    parser.add_argument("--train_dir", type=str,
                        default="../../../train")
    parser.add_argument("--val_dir", type=str,
                        default="../../../test")


    parser.add_argument("--base_model", type=str,
                        default="best_resnet18_rgb.pth")

    parser.add_argument(
        "--class_order",
        type=str,
        default="beach,buildings,forest,harbor,freeway",
    )

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)      
    parser.add_argument("--lr", type=float, default=1e-4)       
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_model", type=str,
                        default="finetuned_resnet18_rgb.pth")


    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")
    print("Using device:", device)
    target_classes = [c.strip() for c in args.class_order.split(",") if c.strip()]
    class_to_idx = {c: i for i, c in enumerate(target_classes)}
    num_classes = len(target_classes)
    print("Class order:", target_classes)
    print("num_classes:", num_classes)
    tf_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    tf_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    def make_subset_dataset(root_dir, transform):
        ds_full = datasets.ImageFolder(root_dir, transform=transform)
        orig_classes = list(ds_full.classes)
        print(f"[{root_dir}] raw classes:", orig_classes)

        new_samples = []
        new_targets = []

        for path, orig_label in ds_full.samples:
            cls_name = orig_classes[orig_label]
            if cls_name not in class_to_idx:
               
                continue
            new_label = class_to_idx[cls_name]
            new_samples.append((path, new_label))
            new_targets.append(new_label)

        ds_full.samples = new_samples
        ds_full.targets = new_targets
        ds_full.classes = target_classes
        ds_full.class_to_idx = class_to_idx
        return ds_full

    train_ds = make_subset_dataset(args.train_dir, tf_train)
    val_ds   = make_subset_dataset(args.val_dir,   tf_eval)

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

    model = build_resnet18(
        num_classes=num_classes,
        pretrained=False,        
        dropout=0.0,
        freeze_backbone=False     
    ).to(device)
    
    load_checkpoint(model, args.base_model, map_location=device)
    print(f"Loaded base model from: {args.base_model}")

    for param in model.parameters():
        param.requires_grad = False

    
    for p in model.fc.parameters():
        p.requires_grad = True
   

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_true, train_pred = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_pred += logits.argmax(1).detach().cpu().tolist()
            train_true += labels.detach().cpu().tolist()

        train_loss = train_loss_sum / len(train_loader)
        train_acc = accuracy_score(train_true, train_pred)
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(imgs)
                preds = logits.argmax(1)

                val_pred += preds.cpu().tolist()
                val_true += labels.cpu().tolist()

        val_acc = accuracy_score(val_true, val_pred)

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}  "
              f"train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}")

       

    save_checkpoint(model, args.out_model)
if __name__ == "__main__":
    main()