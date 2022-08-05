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

"""Functions for synthesis inference."""

import functools
from typing import Sequence

from absl import logging
import gin
import jax
import jax.numpy as jnp
from music_spectrogram_diffusion import audio_codecs
from music_spectrogram_diffusion import event_codec
import numpy as np
import t5x.partitioning
import t5x.utils
import tensorflow as tf


def parse_training_gin_file(gin_file: str, gin_bindings: Sequence[str]) -> str:
  """Parse T5X training files.

  Because of both dynamic_registration, and configuring functions in the main
  binary, the `import __main__ as train_script` needs to be replaced with an
  absolute path, otherwise it tries to configure the current __main__, doesn't
  find it, skips it, and then doesn't dynamically register the other
  functions.

  Could be solved with either having T5X have configurable functions in
  separate libraries from the binary, or not using dynamic_registration
  (explicitly registering each function).

  Args:
    gin_file: Path to config.gin used for training.
    gin_bindings: Additional gin configuration options.

  Returns:
    Parsed gin config string.
  """
  with tf.io.gfile.GFile(gin_file) as f:
    config = f.read()

  config = config.replace('import __main__ as train_script',
                          'import t5x.train as train_script')

  with gin.unlock_config():
    gin.clear_config(clear_constants=True)
    gin.parse_config(config)
    gin.parse_config(gin_bindings)
    gin.finalize()

  logging.info('Parsed gin config: %s', gin.config_str())
  return gin.config_str()


class InferenceModel(object):
  """Wrapper of T5X model for synthesis."""

  def __init__(self, checkpoint_path: str, gin_config: str,
               batch_size: int = 1):
    """Inference model.

    Args:
      checkpoint_path: Checkpoint to restore.
      gin_config: Gin config as a string containing relevant model parameters.
        No additional file loading or parsing is done to this string. Configs
        Should first be processed by parse_training_gin_file above.
      batch_size: To save Puffylite memory and due to the way we process the
        dataset in parallel, this should almost certainly be 1.
    """
    with gin.unlock_config():
      gin.clear_config(clear_constants=True)
      gin.parse_config(gin_config)
      gin.finalize()

    logging.info('Parsed gin config: %s', gin.config_str())

    # Constants.
    self.checkpoint_path = checkpoint_path
    self.batch_size = batch_size
    self.partitioner = t5x.partitioning.PjitPartitioner(
        model_parallel_submesh=(1, 1, 1, 1))

    # Get sequence lengths.
    self.sequence_length = gin.query_parameter('%TASK_FEATURE_LENGTHS')
    self.inputs_length = self.sequence_length['inputs']
    self.targets_length = self.sequence_length['targets']
    self.targets_context_length = self.sequence_length.get(
        'targets_context', None)

    logging.info('Loading model class %s', gin.query_parameter('%MODEL'))
    self.model = gin.get_configurable('MODEL/macro')()

    self.audio_codec: audio_codecs.AudioCodec = gin.get_configurable(
        'AUDIO_CODEC/macro')()
    # The full module path is needed here to disambiguate from the MT3
    # vocabularies module, which will be loaded for transcription inference.
    self.codec: event_codec.Codec = gin.get_configurable(
        'music_spectrogram_diffusion.vocabularies.build_codec')()

  @property
  def input_shapes(self):
    """Model input shapes."""
    shapes = {
        'encoder_input_tokens':
            (self.batch_size, self.inputs_length),
        'decoder_target_tokens':
            (self.batch_size, self.targets_length, self.audio_codec.n_dims)
    }

    # Add fields for context if the model expects them.
    if self.targets_context_length is not None:
      shapes.update({
          'encoder_continuous_inputs':
              (self.batch_size, self.targets_context_length,
               self.audio_codec.n_dims),
          'encoder_continuous_mask':
              (self.batch_size, self.targets_context_length),
      })

    # Add decoder inputs if this is an autoregressive model.
    if 'decoder_input_tokens' in self.model.FEATURE_CONVERTER_CLS.MODEL_FEATURES:
      shapes['decoder_input_tokens'] = shapes['decoder_target_tokens']
    return shapes

  @property
  def input_types(self):
    """Model input types."""
    types = {
        'encoder_input_tokens': np.int32,
        'decoder_target_tokens': np.float32,
    }

    # Add fields for context if the model expects them.
    if self.targets_context_length is not None:
      types.update({
          'encoder_continuous_inputs': np.float32,
          'encoder_continuous_mask': np.int32,
      })

    # Add decoder inputs if this is an autoregressive model.
    if 'decoder_input_tokens' in self.model.FEATURE_CONVERTER_CLS.MODEL_FEATURES:
      types['decoder_input_tokens'] = types['decoder_target_tokens']

    return types

  @functools.lru_cache()
  def _restore_from_checkpoint(self):
    """Restore training state from checkpoint, resets self._predict_fn()."""
    logging.info('Restoring checkpoint %s', self.checkpoint_path)
    train_state_initializer = t5x.utils.TrainStateInitializer(
        optimizer_def=self.model.optimizer_def,
        init_fn=self.model.get_initial_variables,
        input_shapes=self.input_shapes,
        input_types=self.input_types,
        partitioner=self.partitioner)

    restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
        path=self.checkpoint_path, mode='specific', dtype='float32')

    train_state_axes = train_state_initializer.train_state_axes
    train_state = train_state_initializer.from_checkpoint_or_scratch(
        [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0))
    return train_state_axes, train_state

  @property
  def step(self) -> jnp.ndarray:
    _, train_state = self._restore_from_checkpoint()
    return train_state.step

  @functools.lru_cache()
  def _get_predict_fn_and_params(self):
    """Generate a partitioned prediction function for decoding."""
    train_state_axes, train_state = self._restore_from_checkpoint()

    def partial_predict_fn(params, batch, decode_rng):
      return self.model.predict_batch_with_aux(
          params, batch, decode_rng)
    predict_fn = self.partitioner.partition(
        partial_predict_fn,
        in_axis_resources=(
            train_state_axes.params,
            t5x.partitioning.PartitionSpec('data',), None),
        out_axis_resources=t5x.partitioning.PartitionSpec('data',)
    )
    return predict_fn, train_state.params

  def predict(self, batch, seed=0):
    """Predict tokens from preprocessed dataset batch."""
    predict_fn, params = self._get_predict_fn_and_params()
    return predict_fn(params, batch, jax.random.PRNGKey(seed))
