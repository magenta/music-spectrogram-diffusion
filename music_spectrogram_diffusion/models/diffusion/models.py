# Copyright 2023 The Music Spectrogram Diffusion Authors.
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

from typing import Any, Mapping, Optional, Tuple, Union

import clu.metrics
from flax.core import scope as flax_scope
import gin
import jax
import jax.numpy as jnp
from music_spectrogram_diffusion import audio_codecs
from music_spectrogram_diffusion import feature_converters
from music_spectrogram_diffusion import metrics
from music_spectrogram_diffusion.models.diffusion import diffusion_utils
from music_spectrogram_diffusion.models.diffusion import feature_converters as diffusion_feature_converters
from t5x import models

PyTree = Any


class DiffusionModel(
    models.BaseTransformerModel):
  """Encoder-decoder model with continuous outputs."""

  FEATURE_CONVERTER_CLS = feature_converters.ContinuousOutpusEncDecFeatureConverter

  def __init__(self,
               *args,
               diffusion_config=gin.REQUIRED,
               audio_codec=gin.REQUIRED,
               **kwargs):
    self.diffusion_config: diffusion_utils.DiffusionConfig = diffusion_config
    self.audio_codec: audio_codecs.AudioCodec = audio_codec
    super().__init__(*args, **kwargs)

  def _compute_logits(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None) -> jnp.ndarray:
    raise NotImplementedError("Not used for the diffusion model.")

  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, models.Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    input_types = {} if input_types is None else input_types
    encoder_shape = input_shapes["encoder_input_tokens"]
    encoder_type = input_types.get("encoder_input_tokens", jnp.float32)
    decoder_shape = input_shapes["decoder_target_tokens"]
    decoder_type = input_types.get("decoder_target_tokens", jnp.float32)

    initial_variables = self.module.init(
        rng,
        encoder_input_tokens=jnp.ones(encoder_shape, encoder_type),
        decoder_input_tokens=jnp.ones(decoder_shape, decoder_type),
        decoder_noise_time=jnp.ones(decoder_shape[:1], jnp.float32),
        enable_dropout=False)
    return initial_variables

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None
  ) -> Tuple[jnp.ndarray, metrics.MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    # Packing is not supported.
    assert "decoder_segment_ids" not in batch

    targets = batch["decoder_target_tokens"]

    # Clip spectrogram to expected range.
    targets = self.audio_codec.scale_features(
        targets, output_range=[-1., 1.], clip=True)

    if dropout_rng is None:
      # TODO(fjord): This happens during eval, so it's fine to use the same key,
      # but it would probably be good to have it vary by batch.
      dropout_rng = jax.random.PRNGKey(0)

    dropout_rng, diffusion_rng = jax.random.split(dropout_rng)

    z_t, eps, noise_time, include_conditioning = (
        diffusion_utils.get_diffusion_training_input(
            rng=diffusion_rng,
            x0=targets,
            diffusion_config=self.diffusion_config))

    encoder_input_tokens = batch["encoder_input_tokens"]
    encoder_input_tokens *= diffusion_utils.broadcast_to_shape_from_left(
        include_conditioning, encoder_input_tokens.shape)

    model_output = self.module.apply(
        {
            "params": params,
        },
        encoder_input_tokens=encoder_input_tokens,
        decoder_input_tokens=z_t,
        decoder_noise_time=noise_time,
        enable_dropout=True,
        rngs={"dropout": dropout_rng})

    loss = diffusion_utils.calculate_loss(
        x0=targets,
        eps=eps,
        z=z_t,
        time=noise_time,
        model_output=model_output,
        diffusion_config=self.diffusion_config)

    # Loss masking for padding, only relevant during eval because targets are
    # padded during preprocessing for training.
    loss = loss * batch["decoder_target_mask"][..., jnp.newaxis]
    loss = jnp.sum(loss)

    metrics_dict = metrics.compute_base_metrics(
        batch["decoder_target_mask"], loss)
    return loss, metrics_dict

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute log likelihood score on a batch."""
    raise NotImplementedError("Not implemented for diffusion models.")

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = jax.random.PRNGKey(0),
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict by doing a loop over the forward diffusion process.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction.

    Returns:
      A tuple containing:
        the batch of predictions
        an auxiliary dictionary of decoder scores
    """
    # [batch, input_len]
    inputs = batch["encoder_input_tokens"]
    # [batch, target_len]
    target_shape = batch["decoder_target_tokens"].shape

    encodings_and_masks = self.module.apply(
        {"params": params},
        inputs,
        enable_dropout=False,
        method=self.module.encode)

    def pred_fn(z: jnp.ndarray,
                time: jnp.ndarray,
                include_conditioning: bool) -> jnp.ndarray:
      step_encodings_and_masks = jax.tree_map(
          lambda x: x * include_conditioning, encodings_and_masks)
      return self.module.apply(
          {
              "params": params,
          },
          encodings_and_masks=step_encodings_and_masks,
          decoder_input_tokens=z,
          decoder_noise_time=time,
          enable_dropout=False,
          method=self.module.decode)

    if rng is None:
      # TODO(fjord): figure out why this happens.
      rng = jax.random.PRNGKey(0)

    # decodes: [batch_size, len, n_dims]
    pred_x0 = diffusion_utils.eval_scan(
        rng, target_shape, pred_fn, self.diffusion_config)
    decodes = self.audio_codec.scale_to_features(pred_x0, input_range=[-1., 1.])

    # dummy logprobs: [batch_size]
    scores = jnp.zeros((inputs.shape[0],))

    return decodes, scores


class ContextDiffusionModel(
    models.BaseTransformerModel):
  """Encoder-decoder model with continuous outputs."""

  FEATURE_CONVERTER_CLS = diffusion_feature_converters.ContinuousContextFeatureConverter

  def __init__(self,
               *args,
               diffusion_config=gin.REQUIRED,
               audio_codec=gin.REQUIRED,
               **kwargs):
    self.diffusion_config: diffusion_utils.DiffusionConfig = diffusion_config
    self.audio_codec: audio_codecs.AudioCodec = audio_codec
    super().__init__(*args, **kwargs)

  def _compute_logits(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jax.Array] = None) -> jnp.ndarray:
    raise NotImplementedError("Not used for the diffusion model.")

  def get_initial_variables(
      self,
      rng: jax.Array,
      input_shapes: Mapping[str, models.Array],
      input_types: Optional[Mapping[str, jnp.dtype]] = None
  ) -> flax_scope.FrozenVariableDict:
    """Get the initial variables for an encoder-decoder model."""
    initial_variables = self.module.init(
        rng,
        encoder_input_tokens=jnp.ones(
            input_shapes["encoder_input_tokens"],
            input_types["encoder_input_tokens"]),
        encoder_continuous_inputs=jnp.ones(
            input_shapes["encoder_continuous_inputs"],
            input_types["encoder_continuous_inputs"]),
        encoder_continuous_mask=jnp.ones(
            input_shapes["encoder_continuous_mask"],
            input_types["encoder_continuous_mask"]),
        decoder_input_tokens=jnp.ones(
            input_shapes["decoder_target_tokens"],
            input_types["decoder_target_tokens"]),
        decoder_noise_time=jnp.ones(
            input_shapes["decoder_target_tokens"][:1], jnp.float32),
        enable_dropout=False)
    return initial_variables

  def loss_fn(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      dropout_rng: Optional[jnp.ndarray],
      label_smoothing: float = 0.0,
      z_loss: float = 0.0,
      loss_normalizing_factor: Optional[float] = None
  ) -> Tuple[jnp.ndarray, metrics.MetricsMap]:
    """Loss function used for training with a cross-entropy loss."""
    # Packing is not supported.
    assert "decoder_segment_ids" not in batch

    targets = batch["decoder_target_tokens"]
    batch_size = targets.shape[0]

    # Clip spectrogram to expected range.
    targets = self.audio_codec.scale_features(
        targets, output_range=[-1., 1.], clip=True)

    if dropout_rng is None:
      # TODO(fjord): This happens during eval, so it's fine to use the same key,
      # but it would probably be good to have it vary by batch.
      dropout_rng = jax.random.PRNGKey(0)

    dropout_rng, diffusion_rng = jax.random.split(dropout_rng)

    z_t, eps, noise_time, include_conditioning = (
        diffusion_utils.get_diffusion_training_input(
            rng=diffusion_rng,
            x0=targets,
            diffusion_config=self.diffusion_config))

    encoder_input_tokens = batch["encoder_input_tokens"]
    encoder_input_tokens *= diffusion_utils.broadcast_to_shape_from_left(
        include_conditioning, encoder_input_tokens.shape)
    encoder_continuous_mask = batch["encoder_continuous_mask"]
    encoder_continuous_mask *= diffusion_utils.broadcast_to_shape_from_left(
        include_conditioning, encoder_continuous_mask.shape)

    encoder_continuous_inputs = batch["encoder_continuous_inputs"]
    encoder_continuous_inputs = self.audio_codec.scale_features(
        encoder_continuous_inputs, output_range=[-1., 1.], clip=True)

    model_output = self.module.apply(
        {
            "params": params,
        },
        encoder_input_tokens=encoder_input_tokens,
        encoder_continuous_inputs=encoder_continuous_inputs,
        encoder_continuous_mask=encoder_continuous_mask,
        decoder_input_tokens=z_t,
        decoder_noise_time=noise_time,
        enable_dropout=True,
        rngs={"dropout": dropout_rng})

    loss = diffusion_utils.calculate_loss(
        x0=targets,
        eps=eps,
        z=z_t,
        time=noise_time,
        model_output=model_output,
        diffusion_config=self.diffusion_config)

    # Loss masking for padding, only relevant during eval because targets are
    # padded during preprocessing for training.
    loss = loss * batch["decoder_target_mask"][..., jnp.newaxis]
    loss = jnp.sum(loss)

    metrics_dict = metrics.compute_base_metrics(
        batch["decoder_target_mask"], loss)
    metrics_dict["context_frames"] = clu.metrics.Average(
        total=jnp.sum(batch["encoder_continuous_mask"]), count=batch_size)
    return loss, metrics_dict

  def score_batch(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      return_intermediates: bool = False,
  ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Mapping[str, Any]]]:
    """Compute log likelihood score on a batch."""
    raise NotImplementedError("Not implemented for diffusion models.")

  def predict_batch_with_aux(
      self,
      params: PyTree,
      batch: Mapping[str, jnp.ndarray],
      rng: Optional[jax.Array] = jax.random.PRNGKey(0),
  ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Predict by doing a loop over the forward diffusion process.

    Args:
      params: model parameters.
      batch: a batch of inputs.
      rng: an optional RNG key to use during prediction.

    Returns:
      A tuple containing:
        the batch of predictions
        an auxiliary dictionary of decoder scores
    """
    # [batch, target_len]
    target_shape = batch["decoder_target_tokens"].shape

    encoder_continuous_inputs = batch["encoder_continuous_inputs"]
    encoder_continuous_inputs = self.audio_codec.scale_features(
        encoder_continuous_inputs, output_range=[-1., 1.], clip=True)

    encodings_and_masks = self.module.apply(
        {"params": params},
        input_tokens=batch["encoder_input_tokens"],
        continuous_inputs=encoder_continuous_inputs,
        continuous_mask=batch["encoder_continuous_mask"],
        enable_dropout=False,
        method=self.module.encode)

    def pred_fn(z: jnp.ndarray,
                time: jnp.ndarray,
                include_conditioning: bool) -> jnp.ndarray:
      step_encodings_and_masks = jax.tree_map(
          lambda x: x * include_conditioning, encodings_and_masks)
      return self.module.apply(
          {
              "params": params,
          },
          encodings_and_masks=step_encodings_and_masks,
          input_tokens=z,
          noise_time=time,
          enable_dropout=False,
          method=self.module.decode)

    if rng is None:
      # TODO(fjord): figure out why this happens.
      rng = jax.random.PRNGKey(0)

    # decodes: [batch_size, len, n_dims]
    pred_x0 = diffusion_utils.eval_scan(
        rng, target_shape, pred_fn, self.diffusion_config)
    decodes = self.audio_codec.scale_to_features(pred_x0, input_range=[-1., 1.])

    # dummy logprobs: [batch_size]
    scores = jnp.zeros((target_shape[0],))

    return decodes, scores
