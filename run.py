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


import os
import yaml
import argparse
import numpy as np
from collections import OrderedDict
import shutil
from pathlib import Path
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter
import json

import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from models import *
from experiment import VAEXperiment
from dataset import VAEDataset


class Runner:
    def __init__(self, config, del_logs: bool = False):
        self.config = config

        self.log_dirs = [
            "val_recons",
            "val_gt",
            "eval_recons",
            "eval_gt",
            "eval_random_samples",
            "eval_interpolations",
        ]
        self.del_logs = del_logs

    def setup_data(self):
        dataset = self.config["model_params"]["dataset"]
        self.data = VAEDataset(
            dataset,
            **config["data_params"],
            pin_memory=len(self.config["trainer_params"]["gpus"]) != 0,
        )
        self.data.setup()

    def setup_experiment(self, is_hparam_search: bool = False):
        self.model = vae_models[self.config["model_params"]["name"]](
            **self.config["model_params"]
        )
        if "checkpoint_params" in self.config:
            """
            In order to test a loaded model, add the following to the config:
            checkpoint_params:
                path: "path/to/file.ckpt"
            """
            checkpoint = torch.load(self.config["checkpoint_params"]["path"])
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("model.", "")] = v
            self.model.load_state_dict(new_state_dict)

        self.experiment = VAEXperiment(
            self.model,
            self.config["model_params"],
            self.config["exp_params"],
            is_hparam_search=is_hparam_search,
        )

        self.tb_logger = TensorBoardLogger(
            save_dir=self.config["logging_params"]["save_dir"],
            name=self.config["logging_params"]["name"],
        )
        for dir in self.log_dirs:
            Path(f"{self.tb_logger.log_dir}/{dir}").mkdir(exist_ok=True, parents=True)

        # save config
        with open(os.path.join(self.tb_logger.log_dir, "config.json"), "w") as f:
            json.dump(self.config, f)

    def setup_runner(self):
        self.runner = Trainer(
            logger=self.tb_logger,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(
                    save_top_k=1,
                    dirpath=os.path.join(self.tb_logger.log_dir, "checkpoints"),
                    monitor="val_loss",
                    save_last=False,
                ),
            ],
            strategy=DDPPlugin(find_unused_parameters=False),
            enable_progress_bar=False,
            **self.config["trainer_params"],
        )

    def fit_and_test_model(self):
        print(f"======= Training {self.config['model_params']['name']} =======")
        self.runner.fit(self.experiment, datamodule=self.data)

        print(f"======= Testing {self.config['model_params']['name']} =======")
        if "checkpoint_params" in self.config:
            ckpt_path = None
        else:
            ckpt_path = "best"
        self.runner.test(self.experiment, datamodule=self.data, ckpt_path=ckpt_path)

        if self.del_logs:
            self.delete_log_dirs()

    def delete_log_dirs(self):
        for dir in self.log_dirs:
            try:
                shutil.rmtree(f"{self.tb_logger.log_dir}/{dir}")
            except OSError as e:
                print("Error: %s : %s" % (dir, e.strerror))

    def update_config(self, new_config):
        """
        Update corresponding config value with hparam search value
        """
        for group in new_config:
            assert group in self.config.keys(), f"Wrong config group: {group}"
            for key in new_config[group]:
                assert (
                    key in self.config[group].keys()
                ), f"Wrong key: {key} in config group: {group}"
                self.config[group][key] = new_config[group][key]


class RunnerHyperparamWrapper:
    def __init__(self, config, del_logs: bool = False):
        self.hparam_runner = Runner(config, del_logs)

    def search(self, hparam_config):
        # update member config with hparams
        self.hparam_runner.update_config(hparam_config)

        self.hparam_runner.setup_data()
        self.hparam_runner.setup_experiment(is_hparam_search=True)
        self.hparam_runner.setup_runner()

        # run pipeline
        self.hparam_runner.fit_and_test_model()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generic runner for VAE models")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="configs/vae.yaml",
    )
    parser.add_argument(
        "--hsearch",
        help="run hyperparam search",
        type=bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--del_eval",
        help="delete eval runs",
        type=bool,
        default=False,
        required=False,
    )

    args = parser.parse_args()
    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # For reproducibility
    seed_everything(config["exp_params"]["manual_seed"], True)

    if not args.hsearch:
        runner = Runner(config, args.del_eval)
        runner.setup_data()
        runner.setup_experiment()
        runner.setup_runner()
        runner.fit_and_test_model()
    else:
        num_samples = 5
        hparam_config = {
            "model_params": {
                "multi_sample": tune.grid_search(
                    [
                        {"num": 1, "method": "random"},
                        {"num": 2, "method": "random"},
                        {"num": 4, "method": "random"},
                        {"num": 8, "method": "random"},
                    ]
                ),
            },
        }

        hparam_wrapper = RunnerHyperparamWrapper(config, args.del_eval)
        tune.register_trainable("trainable", lambda cfg: hparam_wrapper.search(cfg))

        analysis = tune.run(
            "trainable",
            config=hparam_config,
            resources_per_trial={
                "cpu": 4,
                "gpu": 1,
            },
            metric="interp_fid",
            mode="min",
            num_samples=num_samples,
            progress_reporter=CLIReporter(
                max_progress_rows=200, max_error_rows=200, max_report_frequency=50
            ),
        )
        print(analysis.get_best_config(metric="interp_fid", mode="min"))
