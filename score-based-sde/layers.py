import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.time_proj = nn.Linear(time_emb_dim, out_channels) # Projects time_emb to feature_map channels

    def forward(self, x, t_emb):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # FiLM-like injection: project time_emb and add to feature map
        time_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1) # (B, time_emb_dim) -> (B, out_channels, 1, 1)
        x = x + time_proj # Add projected time embedding to the feature map

        return x

