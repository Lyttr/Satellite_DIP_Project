import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from models.model import build_resnet18, load_checkpoint
from dip.dip_modules import clahe_rgb, bilateral_rgb, unsharp_rgb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str,
                        default='../../collected_data/test')  
    parser.add_argument("--model_path", type=str,
                        default="finetuned_resnet18_rgb.pth")

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--class_order",
        type=str,
        default="beach,buildings,forest,harbor,freeway",
    )

    args = parser.parse_args()

    device = torch.device("cpu")
    print("Using device:", device)


    target_classes = [c.strip() for c in args.class_order.split(",") if c.strip()]
    class_to_idx = {c: i for i, c in enumerate(target_classes)}
    num_classes = len(target_classes)

    tf_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        #transforms.Lambda(lambda im: clahe_rgb(im, clip=2.0, tile=6)), #Accuracy     = 0.6300 F1 (macro)   = 0.6326 AUC (OVR)    = 0.8868
        #transforms.Lambda(lambda im: bilateral_rgb(im,d=3,sigma_color=25,sigma_space=25)), 
        #transforms.Lambda(lambda im: unsharp_rgb(im, k=1, sigma=7)), #Accuracy     = 0.6600F1 (macro)   = 0.6553AUC (OVR)    = 0.8902
        # Accuracy     = 0.4950
        # F1 (macro)   = 0.4723
        # AUC (OVR)    = 0.8477
        #finetune
        # Accuracy     = 0.6200
        # F1 (macro)   = 0.6140
        # AUC (OVR)    = 0.8812
        transforms.ToTensor(),
    ])

    full_test_ds = datasets.ImageFolder(args.test_dir, transform=tf_eval)
    orig_classes = list(full_test_ds.classes)
    print("Raw test classes (from folder):", orig_classes)

    new_samples = []
    new_targets = []

    for path, orig_label in full_test_ds.samples:
        cls_name = orig_classes[orig_label]
        if cls_name in class_to_idx:                
            new_label = class_to_idx[cls_name]      
            new_samples.append((path, new_label))
            new_targets.append(new_label)
    test_ds = full_test_ds
    test_ds.samples = new_samples
    test_ds.targets = new_targets
    test_ds.classes = target_classes
    test_ds.class_to_idx = class_to_idx

    print(f"Filtered test samples: {len(test_ds.samples)}")

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    model = build_resnet18(
        num_classes=num_classes,
        pretrained=False,
        dropout=0.0,
        freeze_backbone=False
    ).to(device)

    load_checkpoint(model, args.model_path, map_location=device)
    model.eval()

    print(f"Loaded model: {args.model_path}")
    y_true, y_pred, y_prob_rows = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().tolist()

            y_true += labels.tolist()
            y_pred += preds
            y_prob_rows += probs.tolist()

    y_prob = np.array(y_prob_rows)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    print("\n===== TEST RESULTS (subset classes) =====")
    print("Class order:", target_classes)
    print(f"#Samples     = {len(y_true)}")
    print(f"Accuracy     = {acc:.4f}")
    print(f"F1 (macro)   = {f1:.4f}")
    print(f"AUC (OVR)    = {auc:.4f}")


if __name__ == "__main__":
    main()