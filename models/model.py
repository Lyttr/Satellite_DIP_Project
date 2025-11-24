
import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
) -> nn.Module:

    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    in_feats = model.fc.in_features
    if dropout and dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, num_classes),
        )
    else:
        model.fc = nn.Linear(in_feats, num_classes)

    head = model.fc[-1] if isinstance(model.fc, nn.Sequential) else model.fc
    nn.init.kaiming_uniform_(head.weight, nonlinearity='relu')
    nn.init.zeros_(head.bias)

    return model


def save_checkpoint(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str, map_location='cpu', strict=True):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model