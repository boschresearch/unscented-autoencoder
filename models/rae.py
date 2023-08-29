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
The model implementation is inspired from the vanilla VAE implementation in: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

import numpy as np
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from torch.distributions.multivariate_normal import MultivariateNormal

from models.models_utils import module_regularization


class RAE(BaseVAE):
    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
    ) -> None:
        super(RAE, self).__init__()

        self.dataset = kwargs["dataset"]

        self.num_channels = in_channels
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
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
        self.fc_z = nn.Linear(hidden_dims[-1], latent_dim)

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

        z = self.fc_z(result)

        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 1024, 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        z = args[2]

        loss_params = kwargs["loss_params"]
        beta = loss_params["beta"]
        dec_reg_params = loss_params["dec_reg"]

        recons_loss = F.mse_loss(recons, input)
        z_reg_loss = torch.mean(torch.linalg.norm(z, dim=1, ord=2))

        # regularize the decoder
        if dec_reg_params == None:
            dec_reg = 0.0
        else:
            decoder_layers = self.get_decoder_layers()

            if "grad_norm" in dec_reg_params["reg_type"]:
                dec_reg_params["grad_input"] = z
                dec_reg_params["grad_output"] = recons

            dec_reg = dec_reg_params["weight"] * module_regularization(
                decoder_layers, dec_reg_params
            )

        loss = recons_loss + beta * z_reg_loss + dec_reg
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "Z_RAE_Loss": z_reg_loss.detach(),
            # "Dec_Reg_Loss": dec_reg.detach(),
        }

    def get_decoder_layers(self) -> List[nn.Module]:
        """
        Returns list of Lin and Conv layers in the entire decoder. Necessary for decoder regularization.
        """
        decoder_layers = []
        decoder_layers.append(self.decoder_input)
        for layer in self.decoder.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                decoder_layers.append(layer)
        for layer in self.final_layer:
            if isinstance(layer, nn.ConvTranspose2d):
                decoder_layers.append(layer)

        return decoder_layers

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        raise RuntimeError("Pure sampling not supported / deterministic model")

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def get_latent_features(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the latent features z
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x D]
        """
        return self.forward(x)[-1]
