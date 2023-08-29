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


from .base import *
from .vanilla_vae import *
from .wae_mmd import *
from .iwae import *
from .fullcov_vae import *
from .rae import *


vae_models = {
    "IWAE": IWAE,
    "WAE_MMD": WAE_MMD,
    "VanillaVAE": VanillaVAE,
    "FullcovVAE": FullcovVAE,
    "RAE": RAE,
}
