
import torch
import torch.nn as nn
from torchvision import models


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return x * attn


class DropBlock2D(nn.Module):
    def __init__(self, block_size=7, drop_prob=0.1):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (
            torch.rand(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device
            )
            < gamma
        ).float()
        mask = nn.functional.max_pool2d(
            mask, self.block_size, stride=1, padding=self.block_size // 2
        )
        return x * (1 - mask)


class EfficientNetB0_2Channel_Fusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        self.reduce_mid = nn.Conv2d(40, 128, 1)
        self.reduce_high = nn.Conv2d(1280, 128, 1)
        self.se = SEBlock(256)
        self.spatial_attn = SpatialAttention()
        self.dropblock = DropBlock2D(block_size=7, drop_prob=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        for i, layer in enumerate(self.model.features):
            x = layer(x)
            if i == 3:
                mid_feat = x
        high_feat = x
        mid_feat = self.reduce_mid(mid_feat)
        high_feat = self.reduce_high(high_feat)
        mid_feat = nn.functional.adaptive_avg_pool2d(mid_feat, high_feat.shape[2:])
        x = torch.cat([mid_feat, high_feat], dim=1)
        x = self.se(x)
        x = self.spatial_attn(x)
        x = self.dropblock(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x


class MultiTaskModel(nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(MultiTaskModel, self).__init__()
        self.base_model = EfficientNetB0_2Channel_Fusion()
        num_features = 256

        self.lesion_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

        self.time_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

        for head in [self.lesion_head, self.time_head]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        for i, layer in enumerate(self.base_model.model.features):
            x = layer(x)
            if i == 3:
                mid_feat = x
        high_feat = x

        mid_feat = self.base_model.reduce_mid(mid_feat)
        high_feat = self.base_model.reduce_high(high_feat)

        mid_feat = nn.functional.adaptive_avg_pool2d(mid_feat, high_feat.shape[2:])
        features = torch.cat([mid_feat, high_feat], dim=1)

        features = self.base_model.se(features)
        features = self.base_model.spatial_attn(features)
        features = self.base_model.dropblock(features)

        features = nn.functional.adaptive_avg_pool2d(features, 1).flatten(1)

        lesion_out = self.lesion_head(features)
        time_out = self.time_head(features)

        return lesion_out, time_out
