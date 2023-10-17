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

"""Synthesis Feature Converters."""

from typing import Mapping

import seqio
import tensorflow as tf


class ContinuousContextFeatureConverter(seqio.FeatureConverter):
  """Feature converter for an encoder-decoder with continuous outputs."""

  TASK_FEATURES = {
      "inputs": seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": seqio.FeatureConverter.FeatureSpec(dtype=tf.float32, rank=2),
      "targets_context": seqio.FeatureConverter.FeatureSpec(
          dtype=tf.float32, rank=2),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.int32),
      "encoder_continuous_inputs":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.float32, rank=2),
      "encoder_continuous_mask":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.bool),
      "decoder_target_tokens":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.float32, rank=2),
      "decoder_target_mask":
          seqio.FeatureConverter.FeatureSpec(dtype=tf.bool),
  }

  def _convert_features(
      self, ds: tf.data.Dataset,
      task_feature_lengths: Mapping[str, int]) -> tf.data.Dataset:
    """Convert the dataset to be fed to the encoder-decoder model.

    The conversion process involves three steps

    1. Each feature in the `task_feature_lengths` is trimmed/padded and
       optionally packed depending on the value of self.pack.
    2. "inputs" fields are mapped to the encoder input and "targets" are mapped
       to decoder input (after being shifted) and target.

    All the keys in the `task_feature_lengths` should be present in the input
    dataset, which may contain some extra features that are not in the
    `task_feature_lengths`. They will not be included in the output dataset.
    One common scenario is the "inputs_pretokenized" and "targets_pretokenized"
    fields.

    Args:
      ds: an input tf.data.Dataset to be converted.
      task_feature_lengths: a mapping from feature to its length.

    Returns:
      ds: the converted dataset.
    """
    def save_lengths(ex):
      for k in ("targets", "targets_context"):
        ex[f"{k}_length"] = tf.shape(ex[k])[0]
      return ex
    ds = ds.map(save_lengths, num_parallel_calls=tf.data.AUTOTUNE)
    ds = self._pack_or_pad(ds, task_feature_lengths)

    def convert_example(
        features: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
      if self.pack:
        # Packing is not supported for 2D features. This doesn't matter for us
        # because target sizes should all equal.
        raise NotImplementedError()

      decoder_target_tokens = features["targets"]
      decoder_target_mask = tf.sequence_mask(
          features["targets_length"],
          maxlen=task_feature_lengths["targets"])

      encoder_continuous_inputs = features["targets_context"]
      encoder_continuous_mask = tf.sequence_mask(
          features["targets_context_length"],
          maxlen=task_feature_lengths["targets_context"])

      d = {
          "encoder_input_tokens": features["inputs"],
          "encoder_continuous_inputs": encoder_continuous_inputs,
          "encoder_continuous_mask": encoder_continuous_mask,
          "decoder_target_tokens": decoder_target_tokens,
          "decoder_target_mask": decoder_target_mask,
      }

      return d

    return ds.map(convert_example, num_parallel_calls=tf.data.AUTOTUNE)

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]) -> Mapping[str, int]:
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    encoder_context_length = task_feature_lengths["targets_context"]
    decoder_length = task_feature_lengths["targets"]

    model_feature_lengths = {
        "encoder_input_tokens": encoder_length,
        "encoder_continuous_inputs": encoder_context_length,
        "encoder_continuous_mask": encoder_context_length,
        "decoder_target_tokens": decoder_length,
        "decoder_target_mask": decoder_length,
    }

    return model_feature_lengths

