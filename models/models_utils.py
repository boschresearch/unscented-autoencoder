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


import copy
from typing import Dict, List

import torch
from torch.autograd import grad
from torch.nn.utils.spectral_norm import spectral_norm


def module_regularization(
    layers: List[torch.nn.Module], reg_params: Dict
) -> torch.Tensor:
    """
    Regularizes a list of modules. Used for decoder regularization in UAE and RAE.
    """
    reg_type = reg_params["reg_type"]

    if reg_type == "weight_decay":
        reg_term = 0.0
        for l in layers:
            reg_term += torch.mean(
                torch.linalg.norm(l.weight.view(l.weight.shape[0], -1), ord=2)
            )
    elif reg_type == "spectral_norm":
        reg_term = 0.0
        for l in layers:
            sn_l = spectral_norm(
                copy.deepcopy(l)
            )  # spectral norm returns the same input layer object, hence copy
            reg_term += torch.mean(
                torch.linalg.norm(sn_l.weight.view(sn_l.weight.shape[0], -1), ord=2)
            )
    elif reg_type == "grad_norm" or "eig_grad_norm":
        """
        grad_output shape [batch_dim, num_image_channels, image_width, image_length]
        grad_input shape [batch_dim, latent_dim]
        """
        output = reg_params["grad_output"].sum((1, 2, 3))
        input = reg_params["grad_input"]

        # compute gradient of output w.r.t input
        reg_term = grad(
            output,
            input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
        )  # grad shape ([batch_dim, latent_dim],)

        # compute norm of the gradient, keep batch dim
        reg_term = torch.linalg.norm(
            reg_term[0], dim=1, ord=2
        )  # [0] since returned as tuple

        # add scaling term
        if reg_type == "eig_grad_norm":
            """
            Scale the decoder gradient penalty term with the largest eigenvalue of the covariance.
            cov: shape [batch_size, latent_dim, latent_dim]
            """
            cov = reg_params["cov"]
            # largest_eigval = torch.linalg.eigvals(cov).real[
            #      :, 0
            # ]  # take first eigval from all batches
            largest_eigval = torch.amax(cov, dim=(1, 2))
            largest_eigval = torch.clamp(largest_eigval, max=100.0)
            reg_term *= largest_eigval

        # mean over batch elements
        reg_term = torch.mean(reg_term)
    else:
        raise ValueError("Invalid regularization: f{reg_type}")

    return reg_term
