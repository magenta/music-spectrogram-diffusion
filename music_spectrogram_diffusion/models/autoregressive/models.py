# Copyright 2022 The Music Spectrogram Diffusion Authors.
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

"""Feature converter and model for continuous inputs."""

import functools
from typing import Any, Callable, Mapping, Optional, Tuple

from flax import jax_utils
import gin
import jax
import jax.numpy as jnp
from music_spectrogram_diffusion import feature_converters
from music_spectrogram_diffusion import metrics
from t5x import models

MetricsMap = metrics.MetricsMap
PyTreeDef = models.PyTreeDef


def continuous_decode(
    inputs: jnp.ndarray,
    cache: Mapping[str, jnp.ndarray],
    tokens_to_logits: Callable[
        [jnp.ndarray, Mapping[str, jnp.ndarray]],
        Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]],
    eos_id: int,
    num_decodes: int = 1,
    decode_rng: jnp.ndarray = jax.random.PRNGKey(42),
    cache_offset: int = 0,
    output_function: Any = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Directly take logits as outputs (spectrograms) without sampling."""
  del eos_id
  del cache_offset
  assert num_decodes == 1

  def scan_fn(carry, x):
    del x
    prev_y, cache, rng = carry
    # print("!!!!!!!!!!!Previous Y:", prev_y.shape)
    outputs, new_cache = tokens_to_logits(prev_y, cache)
    rng, subkey = jax.random.split(rng)
    y = output_function.get_sample(outputs, subkey)
    y = y[:, None, :]  # Add back in time dimension.
    # print("!!!!!!!!!!!New Y:", y.shape)
    carry = (y, new_cache, rng)
    return carry, y

  init_y = inputs[:, :1, :]  # "decoder_input_tokens" Zeros (256, 512, 512).
  init = (init_y, cache, decode_rng)
  (_, decodes) = jax_utils.scan_in_dim(
      scan_fn, init, inputs, axis=1, keepdims=True)

  # decodes: [batch_size, num_decodes=1, len, n_dims]
  # dummy logprobs: [batch_size, num_decodes=1]
  return decodes[:, None], jnp.zeros((inputs.shape[0], 1))


class ContinuousOutputsEncoderDecoderModel(models.EncoderDecoderModel):
  """Encoder-decoder model with continuous outputs."""

  FEATURE_CONVERTER_CLS = feature_converters.ContinuousOutpusEncDecFeatureConverter

  def __init__(self,
               *args,
               output_function=gin.REQUIRED,
               audio_codec=gin.REQUIRED,
               **kwargs):
    self.output_function = output_function
    self.audio_codec = audio_codec
    kwargs["decode_fn"] = functools.partial(continuous_decode,
                                            output_function=output_function)
    super().__init__(*args, **kwargs)

  def loss_fn(
      self,
      params: models.PyTreeDef,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None
  ) -> Tuple[jnp.ndarray, MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    targets = batch["decoder_target_tokens"]
    # TODO(adarob): Rename _compute_logits in T5X to _compute_outputs.
    outputs = self._compute_logits(params, batch, dropout_rng)
    loss = self.output_function.get_loss(outputs, targets)
    # Loss masking for padding, only relevant during eval because targets are
    # padding during preprocessing for training.
    loss = loss * batch["decoder_target_mask"]
    loss = jnp.sum(loss)

    # Generate predicted spectrograms for summaries / metrics.
    # Not used in loss calculation.
    # Teacher forcing, single sampling step so single rng okay.
    rng = jax.random.PRNGKey(42)
    predictions = self.output_function.get_sample(outputs, rng)

    metrics_dict = metrics.compute_base_metrics(
        batch["decoder_target_mask"], loss)
    # Scale to [0, 1].
    predictions = self.audio_codec.scale_features(predictions,
                                                  output_range=[0., 1.0])
    targets = self.audio_codec.scale_features(targets,
                                              output_range=[0., 1.0])

    metrics_dict.update({
        "spectrograms/predicted":
            metrics.ImageSummary(predictions),
        "spectrograms/target":
            metrics.ImageSummary(targets),
    })
    return loss, metrics_dict
