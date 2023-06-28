# SIREN in Equinox (JAX)
Fully-connected neural networks with sinusoidal activation functions converge fast during training and are useful as physics-informed neural networks (PINN) due to the continuous differentiability. [Sinusoidal representation networks (SIREN)](https://arxiv.org/abs/2006.09661) can be used to parametrise any scalar or vector field $\mathbb{R}^n \to \mathbb{R}^m$. This repository contains an implementation of SIREN using [Equinox](https://github.com/patrick-kidger/equinox) (JAX). Essentially, that means a multilayer perceptron (MLP) and special initialisation of weights & biases.

## Getting started
Make sure you have installed [JAX](https://github.com/google/jax#installation) and [Equinox](https://github.com/patrick-kidger/equinox#installation). Copy `src.py` into your project and create an instance of `SIREN` like
```
import jax
from src import SIREN

siren = SIREN(
    num_channels_in=2,                 # n (e.g. image grid)
    num_channels_out=3,                # m (e.g. RGB values)
    num_layers=4,
    num_latent_channels=1024,
    omega=30,                          # angular frequency
    rng_key=jax.random.PRNGKey(420)
)
```

## Training SIREN
For an example on how to train SIREN in Equinox, you can run `main.py`. It additionally requires you to install [Optax](https://github.com/deepmind/optax#installation), [imageio](https://github.com/imageio/imageio) and [tqdm](https://github.com/tqdm/tqdm#installation).
```
python main.py --path_to_image img.py --num_epochs 1000
```
