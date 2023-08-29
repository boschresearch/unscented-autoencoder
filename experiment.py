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


import os, os.path
import numpy as np
from pathlib import Path
from ray import tune
from typing import Dict

import torch
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from sklearn import mixture
from pytorch_fid.fid_score import calculate_fid_given_paths

from models import BaseVAE
from models.types_ import *
from utils import slerpolate


class VAEXperiment(pl.LightningModule):
    def __init__(
        self,
        vae_model: BaseVAE,
        model_params: dict,
        exp_params: dict,
        is_hparam_search: bool = False,
    ) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.model_params = model_params
        self.exp_params = exp_params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.exp_params["retain_first_backpass"]
        except:
            pass
        self.is_hparam_search = is_hparam_search

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def get_loss_params(self) -> Dict:
        loss_params = {}
        if "kld_weight" in self.exp_params:
            loss_params.update(
                {"kld_weight": self.exp_params["kld_weight"]}
            )  # value computed by img.shape[0]/ self.num_train_imgs
        if "frob_norm" in self.exp_params:
            loss_params.update({"frob_norm": self.exp_params["frob_norm"]})
        if "z_reg_weight" in self.exp_params:
            loss_params.update({"beta": self.exp_params["z_reg_weight"]})
        if "dec_reg" in self.exp_params:
            if self.training:
                loss_params.update({"dec_reg": self.exp_params["dec_reg"]})
            else:
                loss_params.update({"dec_reg": None})  # can't compute gradient
        if "sigma_recon" in self.exp_params:
            loss_params.update({"sigma_recon": self.exp_params["sigma_recon"]})

        return loss_params

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, _ = batch

        self.curr_device = real_img.device

        results = self.forward(input=real_img)
        train_loss = self.model.loss_function(
            *results,
            loss_params=self.get_loss_params(),
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, sync_dist=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, _ = batch
        self.curr_device = real_img.device

        results = self.forward(input=real_img)

        val_loss = self.model.loss_function(
            *results,
            loss_params=self.get_loss_params(),
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx,
        )

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()},
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def on_test_end(self):
        self.total_num_images = len(self.trainer.datamodule.eval_dataloader())

        # save gt images
        self.save_gt_images()

        metrics = {}

        # save recon images and compute fid
        self.save_recon_images()
        recon_fid = self.compute_fid_on_datasets(
            f"{self.logger.log_dir}/eval_recons", f"{self.logger.log_dir}/eval_gt"
        )
        metrics.update({"recon_fid": recon_fid})

        # save random samples and compute fid
        train_latent_feats = self.generate_latent_features(
            dataloader=self.trainer.datamodule.train_dataloader(single_sample=False)
        )
        self.save_sampling_eval_images(train_latent_feats)
        sample_fid = self.compute_fid_on_datasets(
            f"{self.logger.log_dir}/eval_random_samples",
            f"{self.logger.log_dir}/eval_gt",
        )
        metrics.update({"sample_fid": sample_fid})

        # save interp images and compute fid
        val_latent_feats = self.generate_latent_features(
            dataloader=self.trainer.datamodule.val_dataloader()
        )
        self.save_interp_eval_images(val_latent_feats)
        interp_fid = self.compute_fid_on_datasets(
            f"{self.logger.log_dir}/eval_interpolations",
            f"{self.logger.log_dir}/eval_gt",
        )
        metrics.update({"interp_fid": interp_fid})

        print("FID ** Reconstruction / Random sample / Interpolation ** FID ")
        print(metrics)

        # store to tensorboard
        self.logger.log_metrics(metrics)

        # store in Ray
        if self.is_hparam_search:
            tune.report(
                recon_fid=recon_fid, sample_fid=sample_fid, interp_fid=interp_fid
            )

        if ("store_posterior", True) in self.exp_params.items():
            post_mean, post_cov = self.generate_posterior()

            eval_post_path = f"{self.logger.log_dir}/eval_posterior"
            Path(eval_post_path).mkdir(exist_ok=True, parents=True)

            torch.save(post_mean[:1000, :], eval_post_path + "/post_mean.pt")
            torch.save(post_cov[:1000, :, :], eval_post_path + "/post_cov.pt")

    def sample_images(self):
        """
        Reconstruction in validation
        """
        dataloader = iter(self.trainer.datamodule.eval_dataloader_shuffle())

        for i in range(100):
            input, _ = next(dataloader)  # [1, C, W, H]
            recon = self.model.generate(input.to(self.curr_device))  # [1, C, W, H]
            assert input.shape[0] == 1
            assert recon.shape[0] == 1

            vutils.save_image(
                recon,
                os.path.join(
                    self.logger.log_dir,
                    "val_recons",
                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}_{i}.png",
                ),
                normalize=True,
            )
            vutils.save_image(
                input,
                os.path.join(
                    self.logger.log_dir,
                    "val_gt",
                    f"gt_{self.logger.name}_Epoch_{self.current_epoch}_{i}.png",
                ),
                normalize=True,
            )

    def save_recon_images(self):
        """
        Saves the reconstructed images from the validation dataset.
        """
        val_dataloader = iter(self.trainer.datamodule.val_dataloader_batch_size_1())
        for i in range(len(val_dataloader)):
            val_input, _ = next(val_dataloader)
            val_input = val_input.to(self.curr_device)
            assert val_input.shape[0] == 1

            recon = self.model.generate(val_input)  # [1, C, H, W]

            vutils.save_image(
                recon.data,
                os.path.join(
                    self.logger.log_dir,
                    "eval_recons",
                    f"recons_{self.logger.name}_img_{i}.png",
                ),
                normalize=True,
            )

    def save_gt_images(self):
        """
        Saves the test images as ground truth in order to compute the FID score.
        """
        eval_dataloader = iter(self.trainer.datamodule.eval_dataloader())

        for i in range(len(eval_dataloader)):
            eval_input, _ = next(eval_dataloader)
            assert eval_input.shape[0] == 1

            vutils.save_image(
                eval_input.data,
                os.path.join(
                    self.logger.log_dir,
                    "eval_gt",
                    f"gt_{self.logger.name}_img_{i}.png",
                ),
                normalize=True,
            )

    def generate_latent_features(self, dataloader):
        latent_feats = []

        dataloader = iter(dataloader)
        for batch in dataloader:
            input, _ = batch
            z = self.model.get_latent_features(input.to(self.curr_device))  # [B, D]
            latent_feats.append(z)

        return torch.cat(latent_feats, dim=0)

    def generate_posterior(self):
        post_mean = []
        post_cov = []

        test_dataloader = iter(self.trainer.datamodule.test_dataloader())
        for batch in test_dataloader:
            input, _ = batch
            mu, cov = self.model.get_posterior(input.to(self.curr_device))

            post_mean.append(mu)
            post_cov.append(cov)

        return (
            torch.cat(post_mean, dim=0).cpu().data.numpy(),
            torch.cat(post_cov, dim=0).cpu().data.numpy(),
        )

    def save_sampling_eval_images(self, latent_feats):
        # fit GMM
        gmm = mixture.GaussianMixture(
            n_components=10, covariance_type="full", max_iter=2000, verbose=2, tol=1e-3
        )
        gmm.fit(latent_feats.cpu())

        # sample and decode
        latent_samples = gmm.sample(self.total_num_images)[0]
        latent_samples = torch.tensor(
            latent_samples, device=self.curr_device, dtype=torch.float
        )

        # split samples into batches so they fit in GPU memory
        split_latent_samples = torch.split(latent_samples, 64, dim=0)
        decoded_images = []
        for batch in split_latent_samples:
            decoded_images.append(self.model.decode(batch))
        decoded_images = torch.cat(decoded_images, dim=0)

        for i in range(decoded_images.shape[0]):
            vutils.save_image(
                decoded_images[i, :, :, :],
                os.path.join(
                    self.logger.log_dir,
                    "eval_random_samples",
                    f"random_sample_{self.logger.name}_img_{i}.png",
                ),
                normalize=True,
            )

    def save_interp_eval_images(self, latent_feats, method="spherical"):
        for i in range(self.total_num_images):
            # take random feature pair and interpolate
            pair_idxs = np.random.choice(list(range(latent_feats.shape[0])), 2)

            if method == "linear":
                interp_feats = (
                    latent_feats[pair_idxs[0], :] + latent_feats[pair_idxs[1], :]
                ) / 2.0
            elif method == "spherical":
                interp_feats = slerpolate(
                    latent_feats[pair_idxs[0], :],
                    latent_feats[pair_idxs[1], :],
                    3,
                )  # interpolate with 3 pts, output shape [num_pts, feat_dim]
                interp_feats = interp_feats[1, :]  # take middle point
            else:
                raise ValueError(f"Invalid interpolation method {method}")

            # decode interpolated features
            interp_image = self.model.decode(interp_feats.unsqueeze(0))

            vutils.save_image(
                interp_image,
                os.path.join(
                    self.logger.log_dir,
                    "eval_interpolations",
                    f"interp_{self.logger.name}_img_{i}.png",
                ),
                normalize=True,
            )

    def compute_fid_on_datasets(self, path1: str, path2: str) -> float:
        """
        Computes FID metric between two datasets in path1 and path2.
        """
        size1 = len([n for n in os.listdir(path1) if os.path.isfile(n)])
        size2 = len([n for n in os.listdir(path2) if os.path.isfile(n)])
        assert (
            size1 == size2
        ), f"The sizes of FID computation datasets do not match: {size1} =/= {size2}"

        return calculate_fid_given_paths(
            paths=[path1, path2],
            batch_size=50,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            dims=2048,
        )

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.exp_params["LR"],
            weight_decay=self.exp_params["weight_decay"],
        )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.exp_params["LR_2"] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model, self.exp_params["submodel"]).parameters(),
                    lr=self.exp_params["LR_2"],
                )
                optims.append(optimizer2)
        except:
            pass

        try:
            if (
                "scheduler_gamma" in self.exp_params
                and self.exp_params["scheduler_gamma"] is not None
            ):
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0], gamma=self.exp_params["scheduler_gamma"]
                )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.exp_params["scheduler_gamma_2"] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1], gamma=self.exp_params["scheduler_gamma_2"]
                        )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
            else:
                scheduler = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                        optims[0],
                        mode="min",
                        factor=0.5,
                        patience=5,
                        verbose=True,
                    ),
                    "monitor": "loss",
                }

                scheds.append(scheduler)

                return optims, scheds

        except:
            return optims
