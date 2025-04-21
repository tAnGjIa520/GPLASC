
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim= 512, latent_dim= 64):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()  # 假设输入在 [0, 1] 范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




class MLP(nn.Module):

    def __init__(self):

        super(MLP, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            self.fc1,
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SupConMLP(nn.Module):
    def __init__(self, feat_dim=500):
        super(SupConMLP, self).__init__()
        self.encoder = MLP()
        self.head = nn.Sequential(
                nn.Linear(500, 500),
                nn.ReLU(inplace=True),
                nn.Linear(500, 500)
            )
    def forward(self, x, return_feat=False):
        encoded = self.encoder(x)
        feat = self.head(encoded)
        if return_feat:
            return feat, encoded
        else:
            return feat

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(500, num_classes)

    def forward(self, features):
        return self.fc(features)
