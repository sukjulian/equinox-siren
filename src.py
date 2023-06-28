import jax.numpy as jnp
import jax
import equinox as eqx
import types
import math


def get_siren_weights_init_fun(omega: float, first_layer: bool = False):

    def init_fun(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):

        fan_in, _ = shape[-2:]
        variance = 1. / fan_in if first_layer else jnp.sqrt(6 / fan_in) / omega

        return jax.random.uniform(key, shape, dtype, minval=-variance, maxval=variance)

    return init_fun


def siren_bias_init(key: jax.random.PRNGKey, shape: tuple, dtype=jnp.float32):

    fan_in = fan_out = shape[-1]
    variance = jnp.sqrt(1. / fan_in)

    return jax.random.uniform(key, (fan_out,), dtype, minval=-variance, maxval=variance)


class MLP(eqx.Module):
    linear_layers: list
    activation: types.FunctionType
    last_activation: types.FunctionType

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        num_layers: int,
        num_latent_channels: int,
        rng_key: jax.random.PRNGKey,
        activation: types.FunctionType = jax.nn.relu,
        plain_last: bool = True
    ):
        rng_sub_keys = jax.random.split(rng_key, num_layers)
        channels_tuple = (num_channels_in, *[num_latent_channels] * (num_layers - 1), num_channels_out)

        self.linear_layers = []
        for num_channels_in, num_channels_out, rng_sub_key in zip(channels_tuple[:-1], channels_tuple[1:], rng_sub_keys):
            self.linear_layers.append(Linear(num_channels_in, num_channels_out, rng_sub_key))

        self.activation = activation
        self.last_activation = lambda x : x if plain_last else activation

    def __call__(self, x: jnp.array):

        for linear_layer in self.linear_layers[:-1]:
            x = self.activation(linear_layer(x))  # x: (batch, num_channels)

        return self.last_activation(self.linear_layers[-1](x))


class Linear(eqx.Module):
    weights: jax.Array
    bias: jax.Array

    def __init__(self, num_channels_in: int, num_channels_out: int, rng_key: jax.random.PRNGKey):
        weights_rng_key, bias_rng_key = jax.random.split(rng_key, 2)

        limit = 1. / math.sqrt(num_channels_in)
        self.weights = jax.random.uniform(weights_rng_key, (num_channels_in, num_channels_out), minval=-limit, maxval=limit)
        self.bias = jax.random.uniform(bias_rng_key, (num_channels_out,), minval=-limit, maxval=limit)

    def __call__(self, x: jnp.array):
        return x @ self.weights + self.bias  # x: (batch, num_channels_in)


class SIREN(eqx.Module):
    omega: float
    mlp: MLP

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        num_layers: int,
        num_latent_channels: int,
        omega: float,
        rng_key: jax.random.PRNGKey,
        plain_last=True
    ):
        mlp_layout = (num_channels_in, num_channels_out, num_layers, num_latent_channels)

        self.omega = omega
        activation = lambda x : jnp.sin(self.omega * x)

        self.mlp = MLP(*mlp_layout, rng_key, activation, plain_last)
        self.init_for_siren(rng_key)

    def init_for_siren(self, rng_key: jax.random.PRNGKey):
        # to understand what is going on here, look at https://docs.kidger.site/equinox/tricks/#custom-parameter-initialisation

        is_linear_layer = lambda layer : isinstance(layer, Linear)

        def get_parameters_lists(module: eqx.Module):
            parameters_lists = {'weights': [], 'biases': []}

            for layer in jax.tree_util.tree_leaves(module, is_leaf=is_linear_layer):
                if is_linear_layer(layer):

                    parameters_lists['weights'].append(layer.weights)
                    parameters_lists['biases'].append(layer.bias)

            return parameters_lists

        is_first_layer = True

        weights_list, biases_list = get_parameters_lists(self.mlp).values()
        init_lists = {'weights': [], 'biases': []}

        for weights, bias, rng_sub_key in zip(weights_list, biases_list, jax.random.split(rng_key, len(weights_list))):
            init_lists['weights'].append(get_siren_weights_init_fun(self.omega, first_layer=is_first_layer)(
                rng_sub_key,
                weights.shape
            ))
            init_lists['biases'].append(siren_bias_init(rng_sub_key, bias.shape))

            is_first_layer = False

        for key in ('weights', 'biases'):
            self.mlp = eqx.tree_at(lambda module : get_parameters_lists(module)[key], self.mlp, replace=init_lists[key])

    def __call__(self, x: jnp.array):
        return self.mlp(x)
