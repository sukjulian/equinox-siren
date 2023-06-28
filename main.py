import jax
from src import SIREN
import optax
import equinox as eqx
import os
import imageio
import jax.numpy as jnp
from tqdm import tqdm

from argparse import ArgumentParser


def main(args):

    siren = SIREN(
        num_channels_in=2,
        num_channels_out=3,
        num_layers=4,
        num_latent_channels=args.num_latent_channels,
        omega=args.siren_omega,
        rng_key=jax.random.PRNGKey(420)
    )

    siren = load_neural_network_parameters("learned_weights.eqx", siren)
    data = load_image(args.path_to_image)

    optimiser = optax.adam(learning_rate=3e-4)  # best learning rate for Adam, hands down
    optimiser_dict = {'object': optimiser, 'state': optimiser.init(eqx.filter(siren, eqx.is_array))}

    siren = train(siren, data, args.num_epochs, optimiser_dict)
    data['values'] = siren(data['grid'])  # inference

    eqx.tree_serialise_leaves("learned_weights.eqx", siren)  # this saves the learned parameters
    save_image(f"{os.path.splitext(args.path_to_image)[0]}_learned.png", data)


def load_image(path):
    image = imageio.v3.imread(path)

    grid = jnp.concatenate((
        jnp.tile(jnp.linspace(0., image.shape[0] / max(image.shape), image.shape[0]), image.shape[1])[..., None],
        jnp.repeat(jnp.linspace(0., image.shape[1] / max(image.shape), image.shape[1]), image.shape[0])[..., None]
    ), axis=-1)

    values = (image.reshape(-1, 3) / 255).astype('f4')

    return {'grid': grid, 'values': values, 'image_dims': image.shape}


def save_image(path, data):
    imageio.imwrite(path, (data['values'].reshape(data['image_dims']) * 255).astype('u1'))


def train(neural_network, data, num_epochs, optimiser_dict):

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        neural_network, optimiser_dict = optimisation_step(neural_network, optimiser_dict, data)

    return neural_network


@eqx.filter_jit
def optimisation_step(neural_network, optimiser_dict, data):
    gradients = compute_loss(neural_network, data)

    neural_network_updates, optimiser_dict['state'] = optimiser_dict['object'].update(gradients, optimiser_dict['state'])
    neural_network = eqx.apply_updates(neural_network, neural_network_updates)

    return neural_network, optimiser_dict


@eqx.filter_grad
def compute_loss(neural_network, data):
    return jnp.mean(jnp.abs(data['values'] - neural_network(data['grid'])))  # mean absolute error


def load_neural_network_parameters(path, neural_network):
    if os.path.exists(path):

        neural_network = eqx.tree_deserialise_leaves(path, neural_network)
        print("Resuming from previously trained neural network weights.")

    return neural_network


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--path_to_image', type=str, default="img.png")
    parser.add_argument('--num_epochs', type=int, default=0)

    parser.add_argument('--num_latent_channels', type=int, default=512)
    parser.add_argument('--siren_omega', type=float, default=24.)

    main(parser.parse_args())
