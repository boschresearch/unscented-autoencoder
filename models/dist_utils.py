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

import torch
import numpy as np
from torch.nn import functional as F


def cov_matrix_from_logvar_corr(
    logvar: torch.Tensor, rho: torch.Tensor
) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return cov_matrix_from_std_corr(std, rho)


def cov_matrix_from_std_corr(std: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """
    Given a vector of standard deviations [sigma_0, sigma_1, ...] and a vector of correlation coefficients [rho_01, rho_02, ...], returns a covariance matrix. Example cov matrix:
    [sigma_0 * sigma_0          | rho_01 * sigma_0 * sigma_1 | rho_02 * sigma_0 * sigma_2
     ------------------------------------------------------------------------------------
     rho_01 * sigma_1 * sigma_0 | sigma_1 * sigma_1          | rho_12 * sigma_0 * sigma_2
     ------------------------------------------------------------------------------------
     rho_02 * sigma_2 * sigma_0 | rho_12 * sigma_2 * sigma_1 | sigma_2 * sigma_2         ]

    :param std: Vector of standard deviations, shape [batch_size, dim]
    :param rho: Vector of correlation coefficients, shape [batch_size, rho_dim]
    :return cov_matrix: Covariance matrix [batch_size, dim, dim]
    """
    assert std.shape[0] == rho.shape[0]
    batch_size = std.shape[0]
    dim = std.shape[-1]
    rho_dim = rho.shape[-1]

    assert rho_dim == dim * (dim - 1) // 2
    assert torch.all(torch.ge(rho, -1 * torch.ones_like(rho))) and torch.all(
        torch.le(rho, torch.ones_like(rho))
    )

    # build symmetric matrix with sigma_x * sigma_x on the diagonal and sigma_x * sigma_y off-diagonal
    var_matrix = std.unsqueeze(-1).repeat(1, 1, dim)
    var_matrix = var_matrix * var_matrix.transpose(2, 1)  # el-wise product

    # build symmetric matrix with ones on diagonal and correlations off-diagonal
    rho_matrix = torch.zeros((batch_size, dim, dim), device=std.device)
    tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
    rho_matrix[:, tril_indices[0], tril_indices[1]] = rho
    rho_matrix = rho_matrix + torch.transpose(rho_matrix, 1, 2)
    rho_matrix = rho_matrix + torch.diag_embed(
        torch.ones(batch_size, dim, device=std.device)
    )
    assert torch.all(
        torch.isclose(rho_matrix, torch.transpose(rho_matrix, 1, 2))
    )  # symmetric

    # build covariance matrix
    cov_matrix = var_matrix * rho_matrix
    assert torch.all(torch.isclose(cov_matrix, cov_matrix.transpose(1, 2)))

    return cov_matrix


def lower_tri_cov_matrix_from_logvar_corr(
    logvar: torch.Tensor, rho: torch.Tensor
) -> torch.Tensor:
    """
    Given a vector of standard deviations [sigma_0, sigma_1, ...] and a vector of correlation coefficients [rho_01, rho_02, ...], returns a lower triangular covariance matrix, example:
    [sigma_0                    | 0                          | 0
     ------------------------------------------------------------------------------------
     rho_01 * sigma_1 * sigma_0 | sigma_1                    | 0
     ------------------------------------------------------------------------------------
     rho_02 * sigma_2 * sigma_0 | rho_12 * sigma_2 * sigma_1 | sigma_2                   ]

    :param std: Vector of standard deviations, shape [batch_size, dim]
    :param rho: Vector of correlation coefficients, shape [batch_size, rho_dim], rho_dim = dim * (dim-1) / 2
    :return lower_tri_cov: Lower triangular covariance matrix [batch_size, dim, dim]
    """
    assert logvar.shape[0] == rho.shape[0]

    std = torch.exp(0.5 * logvar)
    batch_size = std.shape[0]
    dim = std.shape[-1]
    rho_dim = rho.shape[-1]

    assert rho_dim == dim * (dim - 1) // 2
    assert torch.all(torch.ge(rho, -1 * torch.ones_like(rho))) and torch.all(
        torch.le(rho, torch.ones_like(rho))
    )

    # build symmetric matrix with sigma_x * sigma_x on the diagonal and sigma_x * sigma_y off-diagonal
    var_matrix = std.unsqueeze(-1).repeat(1, 1, dim)
    var_matrix = var_matrix * var_matrix.transpose(2, 1)  # el-wise product

    # build symmetric matrix with zeros on diagonal and correlations under
    rho_matrix = torch.zeros((batch_size, dim, dim), device=std.device)
    tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
    rho_matrix[:, tril_indices[0], tril_indices[1]] = rho

    # build lower triangular covariance
    lower_tri_cov = var_matrix * rho_matrix  # multiply correlations and std's
    lower_tri_cov = lower_tri_cov + torch.diag_embed(std)  # add std's on diagonal

    return lower_tri_cov


def get_sigma_pts(mu, lower_tri_cov):
    """
    Computes sigma points.
    """
    batch_size = mu.shape[0]
    feat_dim = mu.shape[1]
    n = 2 * feat_dim + 1
    sigma = torch.empty((batch_size, n, feat_dim), device=mu.device)
    lmd = 1e-3

    sigma[:, 0, :] = mu
    rep_mu = mu.unsqueeze(1).repeat(1, feat_dim, 1)  # repeat mean along rows

    sqrt_scaled_cov = np.sqrt(lmd + feat_dim) * lower_tri_cov

    assert not torch.any(torch.isnan(sqrt_scaled_cov))
    sigma[:, 1 : (feat_dim + 1), :] = rep_mu + sqrt_scaled_cov
    sigma[:, (feat_dim + 1) : 2 * feat_dim + 1, :] = rep_mu - sqrt_scaled_cov

    return sigma


def sample_sigma_from_mu_tril(
    mu: torch.Tensor, tril: torch.Tensor, method: str = "random", num_samples: int = 1
) -> torch.Tensor:
    """
    Takes num_samples sigma points from the distribution defined by (mu, tril). If num_samples is >1, returned batch size is multiplied by num_samples. Arranges multiple sigma points in a 'repeat' fashion, e.g.:
    input original batch: [el1, el2, el3, el4], num_samples=3 (sigmas)
    returned batch of sigmas: [el1_sig1, el2_sig1, el3_sig1, el4_sig1, el1_sig2, el2_sig2, el3_sig2, el4_sig2, el1_sig3, el2_sig3, el3_sig3, el4_sig3]
    """
    # generate sigma points
    sigmas = get_sigma_pts(mu, tril)
    batch_size, num_sigmas, feat_dim = sigmas.shape

    # choose sigma point indices
    if method == "random":
        idxs = torch.tensor(
            np.random.choice([*range(num_sigmas)], size=batch_size * num_samples)
        )  # pick random sigmas, shape [B*S]
    elif method == "random_pairs":
        """
        Samples random pairs of sigma points along an axis.
        """
        assert num_samples % 2 == 0
        # assumes the sigma point ordering is [mean, sigmas in 1/2 hyperspace, - sigmas in 1/2 hyperspace]
        idxs = torch.empty(batch_size * num_samples, dtype=int)
        idxs_first = torch.tensor(
            np.random.choice(
                [*range(1, feat_dim + 1)], size=batch_size * num_samples // 2
            )
        )  # pick first element of the sigma pair, ignore means
        idxs_second = idxs_first + feat_dim  # add corresponding sigma along the axis
        idxs[: batch_size * num_samples // 2] = idxs_first
        idxs[batch_size * num_samples // 2 :] = idxs_second
    elif method == "mean_random_pairs":
        """
        Samples mean and random pairs of sigma points along an axis.
        """
        assert num_samples % 2 == 1
        # assumes the sigma point ordering is [mean, sigmas in 1/2 hyperspace, - sigmas in 1/2 hyperspace]
        idxs_zeros = torch.zeros([batch_size], dtype=torch.long)
        idxs_first = torch.tensor(
            np.random.choice(
                [*range(1, feat_dim + 1)], size=(batch_size, (num_samples - 1) // 2)
            )
        ).flatten()  # pick first element of the sigma pair, ignore means
        idxs_second = idxs_first + feat_dim  # add corresponding sigma along the axis
        idxs = torch.cat([idxs_zeros, idxs_first, idxs_second], dim=0)
    elif method == "top_eigval_pairs" or method == "bottom_eigval_pairs":
        """
        Takes sigma point pairs along the largest or smallest axes, determined by the eigenvalues. Computes an eigval approximation via taking the diagonal elements of tril.
        """
        assert num_samples % 2 == 0
        eigvals = torch.diagonal(tril, dim1=1, dim2=2)
        if method == "bottom_eigval_pairs":
            largest = False
        elif method == "top_eigval_pairs":
            largest = True
        idxs = torch.topk(eigvals, dim=1, k=num_samples // 2, largest=largest)[1]
        idxs = idxs.transpose(1, 0).flatten()  # flatten column-major order
        idxs = torch.cat(
            [idxs + 1, idxs + feat_dim + 1], dim=0
        )  # +1 to consider the 0-index mean, shape [BxS]
    elif method == "mean_top_eigval_pairs" or method == "mean_bottom_eigval_pairs":
        """
        Takes mean and sigma point pairs along the largest or smallest axes, determined by the eigenvalues. Computes an eigval approximation via taking the diagonal elements of tril.
        """
        assert num_samples % 2 == 1
        eigvals = torch.diagonal(tril, dim1=1, dim2=2)
        if method == "bottom_eigval_pairs":
            largest = False
        elif method == "top_eigval_pairs":
            largest = True
        idxs = torch.topk(eigvals, dim=1, k=(num_samples - 1) // 2, largest=False)[1]
        idxs = idxs.transpose(1, 0).flatten()  # flatten column-major order
        zero_idxs = torch.zeros((tril.shape[0]), dtype=torch.long, device=tril.device)
        idxs = torch.cat(
            [zero_idxs, idxs + 1, idxs + feat_dim + 1], dim=0
        )  # +1 to consider the 0-index mean, shape [BxS]
    else:
        raise ValueError("Invalid multi-sample method.")

    # replicate sigmas
    sigmas = sigmas.repeat(num_samples, 1, 1)  # [BxS, 2*D+1, D]

    # index corresponding sigma points
    idxs = idxs.unsqueeze(-1).unsqueeze(-1).long().to(mu.device)
    idxs = idxs.repeat(1, 1, feat_dim)
    sigma_idxd = sigmas.gather(1, idxs).squeeze(1)  # [BxS, D]

    return sigma_idxd


def reparameterize_diagonal(
    mu: torch.Tensor, logvar: torch.Tensor, num_samples: int = 1
) -> torch.Tensor:
    """
    Reparameterization trick to sample num_sample times from N(mu, var) with diagonal covariance.
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :param num_samples: For sampling S times from the Gaussian
    :return: (Tensor) [BxS x D]
    """
    # replicate mu and logvar S times
    mu = mu.repeat(num_samples, 1)  # [BxS, D]
    logvar = logvar.repeat(num_samples, 1)  # [BxS, D]

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def reparameterize_full(
    mu: torch.Tensor, tril: torch.Tensor, num_samples: int = 1
) -> torch.Tensor:
    """
    Reparameterization trick to sample num_sample times from N(mu, LL^T) with full covariance.
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param tril: (Tensor) Lower triangular covariance of the latent Gaussian [B x D x D]
    :param num_samples: For sampling S times from the Gaussian
    :return: (Tensor) [BxS x D]
    """
    # replicate mu and tril S times
    mu = mu.repeat(num_samples, 1)  # [BxS, D]
    tril = tril.repeat(num_samples, 1, 1)  # [BxS, D]

    eps = torch.randn_like(mu)
    prod = torch.matmul(tril, eps.unsqueeze(2))
    return mu + prod.squeeze(2)


def sigma_reconstruction_loss(
    recons_images: torch.Tensor,
    gt_images: torch.Tensor,
    num_sigmas: int,
    method: str = "mean",
):
    """
    Assumes the input images contain multiple samples/sigma points ordered in a repeat-like fashion, e.g. for batch_size=2, num_samples=3:
    [batch_el1_s1, batch_el2_s1, batch_el1_s2, batch_el2_s2, batch_el1_s3, batch_el2_s3]
    Assumes the gt images are not repeated: [gt1, gt2]

    :param recons_images [B*S, C, W, H]
    :param gt_images [B, C, W, H]
    :return loss []
    """
    # compute shapes
    _, num_channels, width, height = recons_images.shape
    flat_img_size = num_channels * width * height
    batch_size = recons_images.shape[0] // num_sigmas

    # reshape to [S, B, C*W*H]
    recons = recons_images.reshape(
        num_sigmas, batch_size, flat_img_size
    )  # [S, B, C*W*H]
    gt = gt_images.reshape(batch_size, flat_img_size)  # [B, C*W*H]

    # compute mean image
    mean = torch.mean(recons, dim=0)  # [B, C*W*H]

    if method == "mean":
        """
        Computes the reconstruction loss only on the mean sigma image.
        || \mu - GT ||^2
        \mu = 1/N \Sum_i sigma_i
        """
        loss = F.mse_loss(mean, gt, reduction="none")  # [B, C*W*H]
    elif method == "mean_var":
        """
        Computes the reconstruction loss only on the mean sigma image and regularizes the image variance.
        || \mu - GT ||^2 + Var**2
        \mu = 1/N \Sum_i sigma_i
        Var = \Sum_i (sigma_i - \mu) ** 2
        """
        var = torch.var(recons, dim=0, unbiased=True) + 1e-6  # [B, C*W*H]
        loss = F.mse_loss(mean, gt, reduction="none") + torch.pow(var, 2)  # [B, C*W*H]
    elif method == "bestof":
        """
        Computes the reconstruction loss only on the best sigma point:
        min_i || sigma_i - GT ||^2
        """
        gt = gt.unsqueeze(0).repeat(num_sigmas, 1, 1)  # [S, B, C*W*H]
        loss = F.mse_loss(recons, gt, reduction="none")  # compute sigma-wise loss
        loss = torch.mean(loss, dim=2)
        best_sigmas = torch.argmin(loss, dim=0)
        loss = torch.take_along_dim(loss, best_sigmas[None, :], dim=0)
    else:
        raise ValueError("Invalid multi-sigma reconstruction method")

    return torch.mean(loss)
