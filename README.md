# Unscented Autoencoder

This is the companion code for the paper Unscented Autoencoder by Faris Janjoš, Lars Rosenbaum, Maxim Dolgov, and J. Marius Zöllner. The paper can be found [here](https://arxiv.org/abs/2306.05256). The code allows the users to reproduce and extend the results reported in the paper. Please cite the above paper when reporting, reproducing or extending the results.

## Technical details

The code is adapted from the following open-source VAE project [PyTorchVAE](https://github.com/AntixK/PyTorch-VAE).

#### Requirements

    Python >= 3.5
    PyTorch >= 1.3
    PyTorch Lightning >= 0.6.0

#### Installation

    $ git clone https://github.com/boschresearch/unscented-autoencoder
    $ cd unscented-autoencoder
    $ pip install -r requirements.txt

#### Datasets
For setting up the CelebA dataset, please refer to [PyTorchVAE](https://github.com/AntixK/PyTorch-VAE). FashionMNIST and CIFAR10 should be downloaded and set up automatically upon first training.

#### Models

 The models included in the repository are implemented in the following files:
- Vanilla VAE / [UAE]((https://arxiv.org/abs/2306.05256)): `models/vanilla_vae.py` <- this base model is used for realizing both the VAE and UAE by setting different config parameters
- [RAE](https://arxiv.org/abs/1903.12436): `models/rae.py`
- [IWAE](https://arxiv.org/abs/1509.00519): `models/iwae.py`
- [WAE-MMD](https://arxiv.org/abs/1711.01558): `models/wae_mmd.py`
- Full covariance models for the VAE / UAE: `models/fullcov_uae.py`

All models training config files are tied to a dataset. Each dataset config set is stored in:
- FashionMNIST: `configs_fashion_mnist/`
- CIFAR10: `configs_cifar10/`
- CelebA: `configs_celeba/`
Each dataset config folder contains the models above.

#### Example usage:
- `python run.py -c configs_cifar10/uae.yaml`: Runs the CIFAR10 training of the full UAE model.
- additional flags:
  - `hsearch=True`, `ray`-based hyperparameter search with config defined in `hsearch_config` in `run.py`
  - `del_eval=True`, remove generated image folders (can get pretty large)

Different models can be realized by setting the following config parameters:
  - `ut_sampling`: sets sampling sigma points or reparameterization trick
  - `sigma_recon`: sets averaging of outputs in a single reconstruction loss (Unscented Transform) or multiple per-sample reconstruction losses (vanilla VAE)
  - `multi_sample`: sets the number of samples and the heuristic
  - `frob_norm`: sets Wasserstein metric or KL divergence
  - `dec_reg`: sets the decoder regularization

For example, in order to train the simplified UT-VAE model (with only the Unscented Transform), set `frob_norm` to `False` and `dec_reg` to `None` in `uae.yaml`.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## License
Unscented Autoencoder is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
