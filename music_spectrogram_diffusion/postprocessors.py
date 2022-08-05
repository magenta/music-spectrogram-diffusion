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

"""Postprocessors."""

import functools
from typing import Any, Callable, Mapping, TypeVar

import librosa
from music_spectrogram_diffusion import audio_codecs
import numpy as np
import tensorflow_hub as hub


T = TypeVar('T')


class TFHubEmbeddingsModel(object):
  """Wrapper around a TF Hub model that computes audio embeddings."""

  def __init__(
      self,
      path: str,
      sample_rate: float,
      model_fn: Callable[[Any, np.ndarray, float], T],
      model_output_to_embeddings_fn: Callable[[T], Mapping[str, np.ndarray]]):
    self._model = hub.load(path)
    self._sample_rate = sample_rate
    self._model_fn = model_fn
    self._model_output_to_embeddings_fn = model_output_to_embeddings_fn

  # Function largely copied from //audio/ears/minimodal/seanet/eval/metrics.py
  def compute_embeddings(
      self,
      samples: np.ndarray,
      sample_rate: float,
  ) -> Mapping[str, np.ndarray]:
    """Compute one or more audio embeddings using a TF Hub model.

    Args:
      samples: 1D numpy array of audio samples.
      sample_rate: Sample rate of `samples`.

    Returns:
      Dictionary of embeddings computed by the model.
    """
    # Avoid `inf` values causing librosa.resample to crash.
    samples = np.nan_to_num(samples)

    # Resample to match the sample rate expected by VGGish.
    # Ideally, we would always avoid resampling (most models operate on 16 kHz).
    samples = librosa.resample(
        samples, orig_sr=sample_rate, target_sr=self._sample_rate)

    # Pad the audio with zeroes to make the audio an integer number of seconds.
    # Note: VGGish uses a window length of 0.96 seconds with no overlap.
    # Padding the audio guarantees the model sees all the audio samples. TRILL
    # is less affected, as it uses a step size of 0.167 seconds.
    length = len(samples)
    target_length = int(np.ceil(length / self._sample_rate)) * self._sample_rate
    padding = target_length - length
    samples = np.pad(
        samples, [padding // 2, padding - padding // 2], mode='constant')

    model_output = self._model_fn(self._model, samples, self._sample_rate)
    return {name: np.array(emb) for name, emb in
            self._model_output_to_embeddings_fn(model_output).items()}


@functools.lru_cache()
def _get_vggish_model():
  return TFHubEmbeddingsModel(
      path='https://tfhub.dev/google/vggish/1',
      sample_rate=16000,
      model_fn=lambda model, y, sr: model(y),
      model_output_to_embeddings_fn=lambda x: {'vggish_embedding': x})


@functools.lru_cache()
def _get_trill_model():
  return TFHubEmbeddingsModel(
      path='https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3',
      sample_rate=16000,
      model_fn=lambda model, y, sr: model(y, sample_rate=sr),
      # model_output_to_embeddings_fn=lambda x: {f'trill_{k}': x[k] for k in x})
      model_output_to_embeddings_fn=lambda x:  # pylint: disable=g-long-lambda
      {f'trill_{k}': x[k] for k in ('embedding',)})


def make_output_dict(audio_codec: audio_codecs.AudioCodec,
                     tokens,
                     example,
                     is_target=False,
                     include_raw_audio=True,
                     use_raw_targets_as_prediction=False):
  """Make targets or predictions a dict with relevant fields."""
  if is_target:
    audio = example['raw_targets']
    output = {'targets': tokens, 'include_raw_audio': include_raw_audio}
    target_keys_to_include = [
        'raw_targets', 'targets_context', 'raw_targets_context',
        'unique_id', 'sequence'
    ]
    for k in target_keys_to_include:
      if k in example:
        output[k] = example[k]
  else:
    if use_raw_targets_as_prediction:
      # This is used for computing baseline metrics or as a metrics sanity check
      audio = example['raw_targets']
    else:
      audio = audio_codec.decode(np.array(tokens, dtype=np.float32)[None, :])[0]
    output = {
        'predictions': tokens,
        'audio': audio,
        'include_raw_audio': include_raw_audio
    }

    prediction_keys_to_include = [
        'unique_id', 'model_timing'
    ]
    for k in prediction_keys_to_include:
      if k in example:
        output[k] = example[k]

  output['embeddings'] = {}
  for model in [_get_vggish_model(), _get_trill_model()]:
    output['embeddings'].update(
        model.compute_embeddings(audio, sample_rate=audio_codec.sample_rate))

  return output
