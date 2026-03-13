import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        # TODO 编码器: 输出分布参数
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu_ = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc_var = nn.Linear(hidden_dim, latent_dim)  # 对数方差
        # TODO 解码器
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim),nn.Sigmoid())

    def encode(self, x):
        h = self.encoder(x) # VAE编码器：将输入x映射到潜在空间的分布参数（均值mu和对数方差log_var）
        return self.fc_mu_(h), self.fc_var(h)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, log_var):
        # 重参数化技巧：从N(0,1)采样，然后变换
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)  # 输出分布参数
        z = self.reparameterize(mu, log_var)  # 采样
        x_recon = self.decode(z)  # 解码重建
        return x_recon, mu, log_var

    def loss_function(self, x_recon, x, mu, log_var):
        # 1. 重建损失
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # 2. KL散度：让后验分布接近标准正态分布
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_loss # 总损失
        return total_loss, recon_loss, kl_loss
