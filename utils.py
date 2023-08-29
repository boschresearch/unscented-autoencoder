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

import pytorch_lightning as pl
import torch


## Utils to handle newer PyTorch Lightning changes from version 0.6


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try:  # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except:  # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


## Evaluation utils
def slerpolate(x, y, num_pts):
    """
    Slerp function adapted from: https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792, first encountered in https://github.com/ParthaEth/Regularized_autoencoders-RAE-/blob/master/my_utility/interpolations.py
    """
    def slerp(val, low, high):
        omega = torch.arccos(
            torch.clip(
                torch.matmul(
                    low / torch.linalg.norm(low), high / torch.linalg.norm(high)
                ),
                -1,
                1,
            )
        )
        so = torch.sin(omega)
        if torch.isclose(so, torch.zeros(1, device=so.device)):
            return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
        return (
            torch.sin((1.0 - val) * omega) / so * low
            + torch.sin(val * omega) / so * high
        )

    alphas = torch.linspace(0, 1.0, num_pts, device=x.device)
    pts = [slerp(alpha, x, y) for alpha in alphas]
    return torch.stack(pts)
