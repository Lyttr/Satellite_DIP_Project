import torch
import torch.nn as nn
from torchvision import models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
        except:
            net = models.resnet18(pretrained=pretrained)
        self.body = nn.Sequential(*list(net.children())[:-1])  # conv1..avgpool
        self.out_dim = net.fc.in_features  # 512

        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.body(x)       # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return x
class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet50(weights=weights)
        except:
            net = models.resnet50(pretrained=pretrained)
        self.body = nn.Sequential(*list(net.children())[:-1]) 
        self.out_dim = net.fc.in_features  # 2048
        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad = False
    def forward(self, x):
        x = self.body(x)         
        x = x.view(x.size(0), -1) 
        return x
class DenseNet121Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.densenet121(weights=weights)
        except:
            net = models.densenet121(pretrained=pretrained)
        self.body = nn.Sequential(
            net.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = net.classifier.in_features  

        if freeze_backbone:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.body(x)            
        x = x.view(x.size(0), -1)   
        return x
class ClassificationHead(nn.Module):
    def __init__(self, in_feats: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        if dropout > 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, num_classes),
            )
        else:
            self.net = nn.Linear(in_feats, num_classes)

        last_linear = self.net[-1] if isinstance(self.net, nn.Sequential) else self.net
        nn.init.kaiming_uniform_(last_linear.weight, nonlinearity="relu")
        nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        return self.net(x)


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained=True, freeze_backbone=False, dropout=0.0):
        super().__init__()
        self.backbone = ResNet18Backbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.head = ClassificationHead(
            in_feats=self.backbone.out_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        feats = self.backbone(x)    # [B,512]
        logits = self.head(feats)   # [B,num_classes]
        return logits


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, map_location="cpu", strict=True):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return model
class DCTCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),        

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), 
        )
        self.fc = nn.Linear(64, out_dim)
        self.out_dim = out_dim

        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)        
        h = h.view(h.size(0), -1) 
        h = self.fc(h)           
        return h
class DualBranchModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        rgb_backbone_type: str = "resnet18",
        pretrained_rgb: bool = True,
        freeze_rgb: bool = True,
        dct_out_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rgb_backbone_type == "resnet18":
            self.rgb_backbone = ResNet18Backbone(
                pretrained=pretrained_rgb,
                freeze_backbone=freeze_rgb,
            ) 
        elif rgb_backbone_type == "resnet50":
            self.rgb_backbone = ResNet50Backbone(
            pretrained=pretrained_rgb,
            freeze_backbone=freeze_rgb,
            )
        elif rgb_backbone_type == "DenseNet121":
            self.rgb_backbone = DenseNet121Backbone(
            pretrained=pretrained_rgb,
            freeze_backbone=freeze_rgb,
            )
        self.dct_backbone = DCTCNN(
            in_channels=1,
            out_dim=dct_out_dim,
        )
        
        fusion_dim = self.rgb_backbone.out_dim + self.dct_backbone.out_dim  # 512 + 128
        self.head = ClassificationHead(
            in_feats=fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, rgb: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_backbone(rgb)   # [B, 512]
        f_dct = self.dct_backbone(dct)   # [B, 128]
        f = torch.cat([f_rgb, f_dct], dim=1)  # [B, 640]
        logits = self.head(f)
        return logits