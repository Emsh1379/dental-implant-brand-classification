import torch.nn as nn
from torchvision import models as tvm

MODELS = ["custom_cnn", "vgg16", "resnet50", "densenet121", "efficientnet_b0", "vit_b_16"]
# (epochs, backbone lr, head lr)
HPARAMS = {"custom_cnn": (30, 1e-3, 1e-3), "vgg16": (12, 1e-4, 1e-3), "resnet50": (12, 2e-4, 1e-3),
           "densenet121": (12, 2e-4, 1e-3), "efficientnet_b0": (12, 5e-4, 2e-3), "vit_b_16": (12, 5e-5, 5e-4)}


class CustomCNN(nn.Module):
    """Small 5-block CNN trained from scratch (baseline)."""
    def __init__(self, n=3):
        super().__init__()
        ch, layers = 3, []
        for o in (32, 64, 128, 192, 256):
            layers += [nn.Conv2d(ch, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                       nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2)]
            ch = o
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(ch, n))

    def forward(self, x):
        return self.head(self.features(x))


def build(name, n=3, pretrained=True):
    w = "DEFAULT" if pretrained else None
    if name == "custom_cnn":
        m = CustomCNN(n); return m, m.head[-1]
    if name == "vgg16":
        m = tvm.vgg16(weights=w); m.classifier[-1] = nn.Linear(4096, n); return m, m.classifier[-1]
    if name == "resnet50":
        m = tvm.resnet50(weights=w); m.fc = nn.Linear(2048, n); return m, m.fc
    if name == "densenet121":
        m = tvm.densenet121(weights=w); m.classifier = nn.Linear(1024, n); return m, m.classifier
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=w); m.classifier[-1] = nn.Linear(1280, n); return m, m.classifier[-1]
    if name == "vit_b_16":
        m = tvm.vit_b_16(weights=w); m.heads.head = nn.Linear(768, n); return m, m.heads.head
    raise ValueError(name)
