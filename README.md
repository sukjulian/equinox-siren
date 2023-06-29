# Rise of the shadow dolphins
[Sinusoidal representation networks (SIREN)](https://arxiv.org/abs/2006.09661) can be used to parametrise any scalar or vector field $\mathbb{R}^n \to \mathbb{R}^m$, converge fast during training and are useful as physics-informed neural networks due to continuous differentiability. This repository contains a JAX implementation of SIREN using [Equinox](https://github.com/patrick-kidger/equinox). Essentially, this means a multilayer perceptron (MLP) and special initialisation of weights & biases.

## Getting started
Make sure you have installed [JAX](https://github.com/google/jax#installation) and [Equinox](https://github.com/patrick-kidger/equinox#installation). Copy `src.py` into your own project. Instances of `SIREN` can be created like
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
For an example on how to train the SIREN in Equinox, look at `main.py`. After installing [Optax](https://github.com/deepmind/optax#installation), [imageio](https://github.com/imageio/imageio) and [tqdm](https://github.com/tqdm/tqdm#installation), you can fit an image `img.png` via
```
python main.py --path_to_image img.png --num_epochs 4000
```
