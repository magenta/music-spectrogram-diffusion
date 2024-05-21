# Copyright 2024 The Music Spectrogram Diffusion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Output distributions for loss functions and sampling."""

# from typing import Any, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class OutputFunction(nn.Module):
  """Base class for functions to sample and evaluate loss."""

  @property
  def expected_num_dims(self):
    raise NotImplementedError

  def get_distribution(self, outputs):
    raise NotImplementedError

  def get_loss(self, outputs, targets):
    p = self.get_distribution(outputs)
    return -p.log_prob(targets)

  def get_sample(self, outputs, seed, sample_shape=()):
    p = self.get_distribution(outputs)
    return p.sample(sample_shape=sample_shape, seed=seed)


class GaussianMixture(OutputFunction):
  """Mixture of Gaussians."""
  n_components: int
  dims_per_component: int
  min_sigma: float = 0.1
  max_sigma: float = 1.0

  @property
  def expected_num_dims(self):
    return self.n_components + 2 * self.n_components * self.dims_per_component

  def get_distribution(self, outputs):
    n_dims = outputs.shape[-1]
    if n_dims != self.expected_num_dims:
      raise ValueError('Output of model and input to GaussianMixture does not'
                       f'have the right number of dims. Given {n_dims}'
                       f'(Full Shape: {outputs.shape}), Expected '
                       f'{self.expected_num_dims}, since mixture has '
                       f'{self.n_components} components, with '
                       f'{self.dims_per_component} dims per component.')

    # Pull out the mixture probabilities first.
    probs = outputs[..., :self.n_components]
    probs = jax.nn.softmax(probs)
    mixture_distribution = tfd.Categorical(probs=probs)
    print('p', probs.shape)

    # Trim from outputs, and continue.
    outputs = outputs[..., self.n_components:]

    # Split mu and sigma.
    n_dims = outputs.shape[-1]
    half = int(n_dims // 2)
    mu = outputs[..., :half]
    sigma = outputs[..., half:]

    print('mu', mu.shape)
    print('sigma', sigma.shape)

    # Reshape new axis for components.
    new_shape = mu.shape[:-1] + tuple([self.n_components, -1])
    mu = jnp.reshape(mu, new_shape)
    sigma = jnp.reshape(sigma, new_shape)

    # Scale sigma.
    sigma = jax.nn.sigmoid(sigma)
    sigma = (self.max_sigma - self.min_sigma) * sigma + self.min_sigma

    print('mu', mu.shape)
    print('sigma', sigma.shape)

    components_distribution = tfd.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)

    return tfd.MixtureSameFamily(
        mixture_distribution=mixture_distribution,
        components_distribution=components_distribution)


class Deterministic(OutputFunction):
  """Deterministic outputs with no sampling."""
  sampling_dither_amount: float = 0.0

  def get_sample(self, outputs, seed, sample_shape=()):
    if self.sampling_dither_amount > 0:
      outputs += (
          jax.random.normal(seed, outputs.shape) * self.sampling_dither_amount)
    return outputs

  def get_loss(self, outputs, targets):
    # For Adafactor, it's okay as long as loss-per-frame is giving equal
    # contributions. So we take the mean across a frame, but then take the
    # sum across batch and time.
    mse = (outputs - targets)**2.0
    loss = jnp.mean(mse, axis=-1)
    return loss
