from layers import DoubleConvBlock
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import math

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], time_emb_dim=256):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # The initial_time_proj was unused and caused an AttributeError. Removed.

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConvBlock(in_channels, feature, time_emb_dim))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConvBlock(features[-1], features[-1]*2, time_emb_dim)

        # Up part of UNET
        # The transposed conv layer is separate from the DoubleConvBlock
        # The DoubleConvBlock after the transposed conv will receive concatenated skip + upsampled features
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2) # Output channels is 'feature'
            )
            # Input channels for the DoubleConvBlock after concatenation will be feature (from upsampled) + feature (from skip) = feature*2
            self.ups.append(DoubleConvBlock(feature*2, feature, time_emb_dim))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # _get_time_embedding needs to be a method of UNet
    def _get_time_embedding(self, t, dim=256):
        # Sinusoidal positional embedding
        # t is a 1D tensor of shape (batch_size,)
        t_ = t.unsqueeze(-1) # (batch_size, 1)
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim)).to(t.device)
        sin_enc = torch.sin(t_ * inv_freq)
        cos_enc = torch.cos(t_ * inv_freq)
        return torch.cat([sin_enc, cos_enc], dim=-1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self._get_time_embedding(t))
        skip_connections = []

        # Downsampling path
        for down_block in self.downs:
            x = down_block(x, t_emb)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Upsampling path
        skip_connections = skip_connections[::-1] # Reverse for proper skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # Transposed Conv (upsampling)
            skip_connection = skip_connections[idx//2] # Get corresponding skip connection

            # Handle size mismatch by center cropping the upsampled feature map
            if x.shape != skip_connection.shape:
                x = transforms.CenterCrop(skip_connection.shape[2:])(x)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip, t_emb) # DoubleConvBlock after concatenation

        return self.final_conv(x)

class VPSDE:
  def __init__(self,beta_min=0.1,beta_max=20.0,T=1000):
    self.beta_min=beta_min
    self.beta_max=beta_max
    self.T=T
  def beta(self,t):
    return self.beta_min+t *(self.beta_max-self.beta_min)/self.T
  def alpha_sq_cum(self,t):
    _t=t/self.T
    return torch.exp(-0.5*_t*(self.beta_min+_t*(self.beta_max-self.beta_min)/2)*self.T)
  def sigma_sq_cum(self,t):
    return 1.0-self.alpha_sq_cum(t)
  def marginal_sde(self,x0,t):
    # Reshape t-dependent scalars for broadcasting
    sqrt_alpha_cum = self.alpha_sq_cum(t).sqrt().view(-1, 1, 1, 1)
    sqrt_sigma_cum = self.sigma_sq_cum(t).sqrt().view(-1, 1, 1, 1)

    mean = sqrt_alpha_cum * x0
    std = sqrt_sigma_cum
    noise=torch.randn_like(x0)
    xt=mean+std*noise
    return xt,std,noise
  def score_fn(self,x,t,noise):
    _,std,_=self.marginal_sde(x,t)
    return -noise/std
