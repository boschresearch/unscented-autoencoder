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

from models.models_utils import module_regularization
from models.dist_utils import (
    reparameterize_diagonal,
    sample_sigma_from_mu_tril,
    sigma_reconstruction_loss,
)


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        dataset: str,
        in_channels: int,
        latent_dim: int,
        ut_sampling: bool = False,
        multi_sample: dict = {"method": "random", "num": 1},
        **kwargs,
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.dataset = dataset
        self.num_channels = in_channels
        self.latent_dim = latent_dim
        self.ut_sampling = ut_sampling
        self.multi_sample_num = multi_sample["num"]
        self.multi_sample_method = multi_sample["method"]
        assert self.multi_sample_method in {
            "random",
            "random_pairs",
            "top_eigval_pairs",
            "bottom_eigval_pairs",
            "mean_random_pairs",
            "mean_top_eigval_pairs",
            "mean_bottom_eigval_pairs",
        }
        if not self.ut_sampling:
            assert (
                self.multi_sample_method == "random"
            ), f"Must use random multi-sampling for ut_sampling: {self.ut_sampling}"
        if (
            self.multi_sample_method == "random_pairs"
            or self.multi_sample_method == "top_eigval_pairs"
            or self.multi_sample_method == "bottom_eigval_pairs"
        ):
            assert self.multi_sample_num % 2 == 0
        elif (
            self.multi_sample_method == "mean_random_pairs"
            or self.multi_sample_method == "mean_top_eigval_pairs"
            or self.multi_sample_method == "mean_bottom_eigval_pairs"
        ):
            assert self.multi_sample_num % 2 == 1

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

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        if not self.ut_sampling:
            # supported sampling methods: random
            z = reparameterize_diagonal(
                mu=mu, logvar=log_var, num_samples=self.multi_sample_num
            )  # [BxS, D]
        else:
            # supported sampling methods: random, random_pairs, top_eigval_pairs, mean_random_pairs, mean_top_eigval_pairs
            z = sample_sigma_from_mu_tril(
                mu=mu,
                tril=torch.diag_embed(torch.exp(0.5 * log_var)),
                method=self.multi_sample_method,
                num_samples=self.multi_sample_num,
            )  # [BxS, D]
        return [
            self.decode(z),
            input,
            mu,
            log_var,
            z,
        ]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]  # [BxS, D]
        input = args[1]  # [B, D]
        mu = args[2]  # [B, D]
        log_var = args[3]  # [B, D]
        z = args[4]  # [BxS, D]

        loss_params = kwargs["loss_params"]
        kld_weight = loss_params["kld_weight"]
        frob_norm = loss_params["frob_norm"]
        dec_reg_params = loss_params["dec_reg"]
        sigma_recon = loss_params["sigma_recon"]

        # compute recons loss of the decoded samples
        if sigma_recon:  # w/ sigma covariance weighting
            assert (
                self.ut_sampling == True
            ), "UT sampling must be on for sigma_recon==True"
            recons_loss = sigma_reconstruction_loss(
                recons_images=recons,
                gt_images=input,
                num_sigmas=self.multi_sample_num,
            )
        else:  # standard loss with equal weighting
            recons_loss = F.mse_loss(
                recons, input.repeat(self.multi_sample_num, 1, 1, 1)
            )

        if frob_norm:  # replace KL div with Frob. norm approx
            std = torch.exp(0.5 * log_var)
            lower_tri_cov = torch.diag_embed(std)

            kld_loss = torch.mean(
                torch.linalg.matrix_norm(
                    lower_tri_cov
                    + 1e-6 * torch.ones_like(lower_tri_cov)  # +eps for stability
                    - torch.diag_embed(
                        torch.ones(
                            lower_tri_cov.shape[0],
                            lower_tri_cov.shape[1],
                            device=lower_tri_cov.device,
                        )
                    ),
                    ord="fro",
                )
                + torch.linalg.norm(mu, dim=1, ord=2)
            )
        else:
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
            )

        # regularize the decoder
        if dec_reg_params == None:
            dec_reg = 0.0
        else:
            assert not dec_reg_params["reg_type"] == "weight_decay"

            if "grad_norm" in dec_reg_params["reg_type"]:
                dec_reg_params["grad_input"] = mu
                dec_reg_params["grad_output"] = self.decode(mu)
                if dec_reg_params["reg_type"] == "eig_grad_norm":
                    dec_reg_params["cov"] = torch.diag_embed(torch.exp(log_var))

            dec_reg = dec_reg_params["weight"] * module_regularization(
                layers=None, reg_params=dec_reg_params
            )

        loss = recons_loss + kld_weight * kld_loss + dec_reg
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image from the mean latent vector
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

    def get_posterior(self, x: Tensor) -> List[Tensor]:
        """
        Given an input image x, returns the aggregated posterior mean and covariance matrix
        :param x: (Tensor) [B x C x H x W]
        :return mu: (Tensor) [B x D]
        :return cov: (Tensor) [B x D x D]
        """
        _, _, mu, log_var, _ = self.forward(x)
        var = torch.exp(log_var)
        cov = torch.diag_embed(var)
        return mu, cov
