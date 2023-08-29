""" Unscented Autoencoder (ICML 2023).
Copyright (c) 2023 Robert Bosch GmbH
@author: Faris Janjos
@author: Lars Rosenbaum
@author: Maxim Dolgov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
The model implementation is adapted from the IWAE implementation in: https://github.com/AntixK/PyTorch-VAE/blob/master/models/iwae.py
"""

import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class IWAE(BaseVAE):
    def __init__(
        self,
        dataset: str,
        in_channels: int,
        latent_dim: int,
        num_samples: int = 1,
        **kwargs
    ) -> None:
        super(IWAE, self).__init__()

        self.dataset = dataset
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.num_channels = in_channels

        modules = []
        hidden_dims = [32, 64, 128, 256, 512, 1024]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 8 * 8)

        dec_hidden_dims = [1024, 512, 256]
        if self.dataset == "celeba":
            dec_hidden_dims.append(128)

        for i in range(len(dec_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        dec_hidden_dims[i],
                        dec_hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(dec_hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                dec_hidden_dims[-1],
                out_channels=self.num_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (Tensor) [BS x x D]
        :return: (Tensor) [BS x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 1024, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)  # [BS x C x H x W ]
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = self.reparameterize(mu, log_var)  # [B x S x D]
        eps = (z - mu) / log_var  # Prior samples
        B, _, _ = z.shape
        return [
            self.decode(z.reshape(B * self.num_samples, self.latent_dim)),
            input,
            mu,
            log_var,
            z,
            eps,
        ]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        eps = args[5]

        recons = recons.view(
            [z.size(0), -1, recons.size(1), recons.size(2), recons.size(3)]
        )  # [B x S x C x H x W]
        input = input.repeat(self.num_samples, 1, 1, 1, 1).permute(
            1, 0, 2, 3, 4
        )  # [B x S x C x H x W]

        loss_params = kwargs["loss_params"]
        kld_weight = loss_params["kld_weight"]

        log_p_x_z = (
            ((recons - input) ** 2).flatten(2).mean(-1)
        )  # Reconstruction Loss [B x S]
        kld_loss = -0.5 * torch.sum(
            1 + log_var - mu**2 - log_var.exp(), dim=2
        )  ## [B x S]
        # Get importance weights
        log_weight = log_p_x_z + kld_weight * kld_loss  # .detach().data

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim=-1)
        # kld_loss = torch.mean(kld_loss, dim = 0)

        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0)

        return {
            "loss": loss,
            "Reconstruction_Loss": log_p_x_z.mean(),
            "KLD": -kld_loss.mean(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z).squeeze()
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image from the mean latent.
        Returns only the first reconstructed sample
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decode(self.encode(x)[0])

    def get_latent_features(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the latent features of the mean
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x D]
        """
        return self.encode(x)[0]
