"""Two landmark networks that share the same HRNet backbone (timm, ImageNet-pretrained).

HRNetLandmarkNet (proposed)   heatmap regression: one Gaussian heatmap per landmark, decoded by
                              argmax. Literature: Pfister et al. (2015) and Payer et al. (2019)
                              argue that predicting spatially structured heatmaps beats predicting
                              absolute coordinates, because the output keeps its position
                              information and gets dense supervision.

CoordRegressionNet (baseline) direct coordinate regression: the same backbone, but the head outputs
                              the 8 coordinates themselves. This is the baseline the heatmap papers
                              compare against. Only the output representation differs, so any gap
                              between the two models is attributable to that choice.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetLandmarkNet(nn.Module):
    output_type = "heatmap"

    def __init__(self, backbone="hrnet_w32", pretrained=True, num_keypoints=4,
                 input_size=384, stride=4, head_channels=256):
        super().__init__()
        self.input_size, self.stride = input_size, stride
        self.target = input_size // stride
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        in_ch = sum(self.backbone.feature_info.channels())
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_keypoints, kernel_size=1),
        )

    def forward(self, x):
        size = (self.target, self.target)
        feats = self.backbone(x)
        feats = [f if f.shape[-2:] == size
                 else F.interpolate(f, size=size, mode="bilinear", align_corners=False)
                 for f in feats]
        x = self.fuse(torch.cat(feats, dim=1))
        return self.head(x)  # (N, K, H/stride, W/stride), linear output


class CoordRegressionNet(nn.Module):
    """Baseline: output (N, K, 2) normalised coordinates (fraction of the input size).

    head="flatten" (default) reduces the stride-32 features with a 1x1 convolution and flattens
    them, so the fully connected layer still sees WHERE features are. head="gap" averages the
    features over space first, which removes the position information. The gap variant is the
    weaker, classic formulation and is kept for an ablation."""
    output_type = "coords"

    def __init__(self, backbone="hrnet_w32", pretrained=True, num_keypoints=4,
                 input_size=384, head="flatten"):
        super().__init__()
        self.input_size, self.num_keypoints = input_size, num_keypoints
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        info = self.backbone.feature_info
        reductions, channels = list(info.reduction()), list(info.channels())
        self.feat_idx = reductions.index(32) if 32 in reductions else len(reductions) - 1
        self.stride = reductions[self.feat_idx]
        ch = channels[self.feat_idx]
        if head == "gap":
            self.reduce = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Linear(ch, 512), nn.ReLU(inplace=True), nn.Dropout(0.1))
        elif head == "flatten":
            fs = input_size // self.stride
            self.reduce = nn.Sequential(
                nn.Conv2d(ch, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Flatten(), nn.Linear(64 * fs * fs, 512), nn.ReLU(inplace=True), nn.Dropout(0.1))
        else:
            raise ValueError(f"unknown regression head: {head}")
        self.fc = nn.Linear(512, num_keypoints * 2)

    def forward(self, x):
        f = self.backbone(x)[self.feat_idx]
        return self.fc(self.reduce(f)).view(-1, self.num_keypoints, 2)


def build_model(cfg, pretrained=None):
    m, d = cfg["model"], cfg["data"]
    pt = m["pretrained"] if pretrained is None else pretrained
    if m.get("type", "heatmap") == "regression":
        return CoordRegressionNet(backbone=m["backbone"], pretrained=pt,
                                  num_keypoints=m["num_keypoints"], input_size=d["input_size"],
                                  head=m.get("head", "flatten"))
    return HRNetLandmarkNet(
        backbone=m["backbone"],
        pretrained=pt,
        num_keypoints=m["num_keypoints"],
        input_size=d["input_size"],
        stride=d["heatmap_stride"],
        head_channels=m.get("head_channels", 256),
    )
