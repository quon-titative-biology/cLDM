############################################### 1. Define the module ##############################################
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random


# reproducibility
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.enabled = False
     #torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


# initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])
    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)
    def mode(self):
        return self.mean


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, hidden_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm2d(hidden_channels),
                                      nn.ReLU(True),
                                      nn.Conv2d(hidden_channels, out_channels,
                                                kernel_size=1, padding=1, bias=False),
                                      nn.BatchNorm2d(out_channels))
    def forward(self, input):
        residual = input
        output = residual + self.resblock(residual)
        return output


class ResidualStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_res_layers: int):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_channels, hidden_channels, out_channels)] * n_res_layers)
    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


############################################### 2. Define the model ###############################################
class ResVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 hidden_dims: list = [16, 32, 64],
                 latent_dim: int = 3,
                 n_res_layers: int = 2) -> None:
        super(ResVAE, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.quant_conv = torch.nn.Conv2d(2*latent_dim, 2*latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_dim, latent_dim, 1)
        modules = []
        # Build Encoder
        # Resizing Block [input image --> 196 x 196]
        ResizeBlock = nn.Sequential(
                      nn.Conv2d(in_channels, hidden_dims[0],    # 3 --> 8
                                kernel_size=5, stride=(1,2), padding=0),
                      nn.BatchNorm2d(hidden_dims[0]),
                      nn.LeakyReLU(),
                      nn.Conv2d(hidden_dims[0], hidden_dims[1], # 8 --> 16
                                kernel_size=4, stride=(1,2), padding=0),
                      nn.BatchNorm2d(hidden_dims[1]),
                      nn.LeakyReLU()
                      )
        modules.append(ResizeBlock)
        modules.append(nn.Sequential(                           # 16 --> 32
                       nn.Conv2d(hidden_dims[1], hidden_dims[-1],
                                 kernel_size=5, stride=2, padding=1),
                       nn.BatchNorm2d(hidden_dims[-1]),
                       nn.LeakyReLU())
                      )
        # Residual Blocks
        # modules.append(ResidualStack(hidden_dims[-1], hidden_dims[-1], hidden_dims[-1], n_res_layers)) # 32 --> 32
        self.encoder = nn.Sequential(*modules)
        self.conv_out = nn.Sequential(nn.Conv2d(hidden_dims[-1],
                                                2*latent_dim,
                                                kernel_size=4,
                                                stride=2,
                                                padding=1),
                                      nn.BatchNorm2d(2*latent_dim),
                                      nn.LeakyReLU())
        # Build Decoder
        self.conv_in = nn.Sequential(nn.BatchNorm2d(latent_dim),
                                     nn.LeakyReLU(),
                                     nn.ConvTranspose2d(latent_dim,
                                                        hidden_dims[-1],
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        output_padding=0))
        modules = []
        # Residual Blocks
        # modules.append(ResidualStack(hidden_dims[-1], hidden_dims[-1], hidden_dims[-1], n_res_layers)) # 32 --> 32
        modules.append(nn.Sequential(
                       nn.BatchNorm2d(hidden_dims[-1]),
                       nn.LeakyReLU(),
                       nn.ConvTranspose2d(hidden_dims[-1],
                                          hidden_dims[1],
                                          kernel_size=3,
                                          stride=2,
                                          padding=0,
                                          output_padding=0),
                       nn.BatchNorm2d(hidden_dims[1]),
                       nn.LeakyReLU()))
        modules.append(nn.Sequential(
                       nn.ConvTranspose2d(hidden_dims[1],
                                          hidden_dims[0],
                                          kernel_size=4,
                                          stride=2,
                                          padding=0,
                                          output_padding=0),
                       nn.BatchNorm2d(hidden_dims[0]),
                       nn.LeakyReLU()))
        modules.append(nn.Sequential(
                       nn.ConvTranspose2d(hidden_dims[0],
                                          in_channels,
                                          kernel_size=5,
                                          stride=(1,2),
                                          padding=0,
                                          output_padding=(0,1)),
                       nn.BatchNorm2d(in_channels),
                       nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                           nn.ConvTranspose2d(in_channels,
                                              in_channels,
                                              kernel_size=4,
                                              stride=(1,2),
                                              padding=0,
                                              output_padding=0),
                           nn.Tanh())
    def encode(self, input):
        h = self.encoder(input)
        h = self.conv_out(h)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior
    def decode(self, z):
        z = self.post_quant_conv(z)
        z = self.conv_in(z)
        decoded = self.decoder(z)
        rescontruction = self.final_layer(decoded)
        return rescontruction
    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        rescontruction = self.decode(z)
        return rescontruction, posterior


#Training model
def vae_training_mode(model):
    model.train()
    print("...")


def vae_eval_mode(model):
    model.eval()
    print("...")


def get_loss(model, x, type="mse"):
    reconstruction, posterior = model(x)
    kl_divergence = posterior.kl().sum()
    if type != "mse":
       reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    else:
       reconstruction_loss = F.mse_loss(reconstruction, x, reduction='sum')
    loss = reconstruction_loss + kl_divergence
    return loss


