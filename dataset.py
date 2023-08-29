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
import torch
import copy
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import transforms, datasets
from torchvision.datasets import CelebA
import zipfile


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        upsample: int = 1,
        pin_memory: bool = False,
        replicate_data: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.dataset = dataset
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.replicate_data = replicate_data

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  MNIST Dataset  =========================
        if self.dataset == "mnist":
            mnist_train_data = datasets.MNIST(
                root="./datasets/mnist_data/",
                train=True,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.ToTensor()]
                ),
                download=True,
            )
            mnist_len = len(mnist_train_data)
            self.train_dataset = Subset(mnist_train_data, range(10000, mnist_len))
            self.val_dataset = Subset(mnist_train_data, range(0, 10000))
            self.test_dataset = datasets.MNIST(
                root="./datasets/mnist_data/",
                train=False,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.ToTensor()]
                ),
                download=True,
            )
        #       =========================  Fashion MNIST Dataset  =========================
        if self.dataset == "fashion_mnist":
            fashion_mnist_train_data = datasets.FashionMNIST(
                root="./datasets/fashion_mnist_data/",
                train=True,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.ToTensor()]
                ),
                download=True,
            )
            fashion_mnist_len = len(fashion_mnist_train_data)
            self.train_dataset = Subset(
                fashion_mnist_train_data, range(10000, fashion_mnist_len)
            )
            self.val_dataset = Subset(fashion_mnist_train_data, range(0, 10000))
            self.test_dataset = datasets.FashionMNIST(
                root="./datasets/fashion_mnist_data/",
                train=False,
                transform=transforms.Compose(
                    [transforms.Resize(32), transforms.ToTensor()]
                ),
                download=True,
            )
        #       =========================  CIFAR10 dataset  =========================
        elif self.dataset == "cifar10":
            cifar10_train_data = datasets.CIFAR10(
                root="./datasets/cifar10_data/",
                train=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
                download=True,
            )
            cifar10_len = len(cifar10_train_data)
            self.train_dataset = Subset(cifar10_train_data, range(10000, cifar10_len))
            self.val_dataset = Subset(cifar10_train_data, range(0, 10000))
            self.test_dataset = datasets.CIFAR10(
                root="./datasets/cifar10_data/",
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                ),
                download=True,
            )
        #       =========================  CelebA Dataset  =========================
        elif self.dataset == "celeba":
            celeba_transforms = transforms.Compose(
                [
                    transforms.CenterCrop(148),
                    transforms.Resize(self.patch_size),
                    transforms.ToTensor(),
                ]
            )
            self.train_dataset = MyCelebA(
                self.data_dir,
                split="train",
                transform=celeba_transforms,
                download=False,
            )
            self.val_dataset = MyCelebA(
                self.data_dir,
                split="valid",
                transform=celeba_transforms,
                download=False,
            )
            self.test_dataset = MyCelebA(
                self.data_dir,
                split="test",
                transform=celeba_transforms,
                download=False,
            )
        else:
            raise ValueError(f"invalid dataset {self.dataset}")

    #       ===============================================================

    def train_dataloader(self, single_sample=False) -> DataLoader:
        replicated_dataset = self.train_dataset
        if not single_sample:
            replicated_dataset = ConcatDataset(
                [
                    self.train_dataset,
                    *[
                        copy.deepcopy(self.train_dataset)
                        for _ in range(self.replicate_data - 1)
                    ],
                ]
            )

        return DataLoader(
            replicated_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def val_dataloader_batch_size_1(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def eval_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def eval_dataloader_shuffle(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
