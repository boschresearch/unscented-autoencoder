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


import matplotlib.pyplot as plt
import pytest
import torch

from utils import slerpolate


def test_slerpolate():
    x = torch.tensor([10.0, 0.0])
    y = torch.tensor([0.0, 1.0])
    C = torch.eye(2)
    zs = slerpolate(x, y, C, 11)

    zs_normal = slerpolate(x, y, None, 11)

    torch.testing.assert_close(zs, zs_normal)

    plt.figure()
    plt.plot(*x, color="r", marker="*")
    plt.plot(*y, color="g", marker="*")
    plt.plot(*torch.transpose(zs, 0, 1), color="b", marker="o", alpha=0.2)
    plt.axis("equal")
    plt.show()
    plt.grid(True)
    plt.close()
