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

"""Audio codecs that compute features and decode back to audio."""

from typing import Optional

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# Hack for caching SavedModels in the module to avoid reloading.
_MODEL_CACHE = {
    'melgan': None,
}


_MODEL_PATHS = {
    'melgan': 'https://tfhub.dev/google/soundstream/mel/decoder/music/1',
}


def _load_model_from_cache(key):
  """Loads a model from the cache, optionally adds model to cache."""
  if _MODEL_CACHE[key] is None:
    _MODEL_CACHE[key] = hub.load(_MODEL_PATHS[key])
  return _MODEL_CACHE[key]


class Audio2Mel(tf.keras.Model):
  """Audio2Mel."""

  def __init__(self,
               sample_rate: Optional[int] = 16000,
               n_fft: int = 1024,
               hop_length: int = 160,
               win_length: int = 400,
               n_mel_channels: Optional[int] = 64,
               drop_dc: bool = True,
               mel_fmin: float = 60.0,
               mel_fmax: float = 7800.0,
               clip_value_min: float = 1e-5,
               clip_value_max: float = 1e8,
               log_amplitude: bool = True):
    """Builds the Audio2Mel frontend.

    Args:
      sample_rate: sampling rate. Need to be provided if `n_mel_channels` is not
        `None`.
      n_fft: length of the FFT, in samples.
      hop_length: length of the hop size, in samples.
      win_length: length of the window, in samples.
      n_mel_channels: number of mel channels. If set to None, will return the
        full magnitude STFT.
      drop_dc: if `True`, drop the STFT DC coefficient. Used only when
        n_mel_channels is `None`.
      mel_fmin: lowest frequency in the mel filterbank in Hz.
      mel_fmax: highest frequency in the mel filterbank in Hz.
      clip_value_min: minimal value of the (mel)-spectrogram before log. Used
        only when `log_amplitude` is `True`.
      clip_value_max: maximal value of the (mel)-spectrogram before log. Used
        only when `log_amplitude` is `True`.
      log_amplitude: if `True` apply log amplitude scaling.
    """

    super().__init__()

    self._n_fft = n_fft
    self._hop_length = hop_length
    self._win_length = win_length
    self._sample_rate = sample_rate
    self._clip_value_min = clip_value_min
    self._clip_value_max = clip_value_max
    self._log_amplitude = log_amplitude
    self._n_mel_channels = n_mel_channels
    self._drop_dc = drop_dc

    if n_mel_channels is None:
      self.mel_basis = None
    else:
      if sample_rate is None:
        raise ValueError(
            '`sample_rate` must be provided when `n_mel_channels` is not `None`'
        )
      if mel_fmax is None:
        mel_fmax = sample_rate // 2

      self.mel_basis = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins=n_mel_channels,
          num_spectrogram_bins=n_fft // 2 + 1,
          sample_rate=sample_rate,
          lower_edge_hertz=mel_fmin,
          upper_edge_hertz=mel_fmax)

  def call(self, audio, training=False):
    """Computes the mel spectrogram of the input audio samples.

    Coefficients are clipped before log compression to avoid log(0) and large
    coefficients.

    Args:
      audio: input sample of shape (batch_size, num_samples).
      training: flag to distinguish between train and test time behavior.

    Returns:
      Mel spectrogram of shape (batch_size, time_frames, freq_bins).
    """
    fft = tf.signal.stft(
        audio,
        frame_length=self._win_length,
        frame_step=self._hop_length,
        fft_length=self._n_fft,
        window_fn=tf.signal.hann_window,
        pad_end=True)
    fft_modulus = tf.abs(fft)

    if self.mel_basis is not None:
      output = tf.matmul(fft_modulus, self.mel_basis)
    else:
      output = fft_modulus
      if self._drop_dc:
        output = output[:, :, 1:]

    if self._log_amplitude:
      output = tf.clip_by_value(
          output,
          clip_value_min=self._clip_value_min,
          clip_value_max=self._clip_value_max)
      output = tf.math.log(output)
    return output


class AudioCodec(object):
  """Base class for audio codec that encodes features and decodes to audio."""

  name: str
  n_dims: int
  sample_rate: int
  hop_size: int
  min_value: float
  max_value: float
  pad_value: float
  additional_frames_for_encoding: int = 0

  @property
  def abbrev_str(self):
    return self.name

  @property
  def frame_rate(self):
    return int(self.sample_rate // self.hop_size)

  def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
    """Linearly scale features to network outputs range."""
    min_out, max_out = output_range
    if clip:
      features = jnp.clip(features, self.min_value, self.max_value)
    # Scale to [0, 1].
    zero_one = (features - self.min_value) / (self.max_value - self.min_value)
    # Scale to [min_out, max_out].
    return zero_one * (max_out - min_out) + min_out

  def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
    """Invert by linearly scaling network outputs to features range."""
    min_out, max_out = input_range
    outputs = jnp.clip(outputs, min_out, max_out) if clip else outputs
    # Scale to [0, 1].
    zero_one = (outputs - min_out) / (max_out - min_out)
    # Scale to [self.min_value, self.max_value].
    return zero_one * (self.max_value - self.min_value) + self.min_value

  def encode(self, audio):
    """Encodes audio to features."""
    raise NotImplementedError

  def decode(self, features):
    """Decodes features to audio."""
    raise NotImplementedError

  def to_images(self, features):
    """Maps a batch of features to images for visualization."""
    assert features.ndim == 3
    return self.scale_features(features, output_range=(0.0, 1.0))

  @property
  def context_codec(self):
    """Codec for encoding audio context."""
    return self


class MelGAN(AudioCodec):
  """Invertible Mel Spectrogram with 128 dims and 16kHz."""

  name = 'melgan'
  n_dims = 128
  sample_rate = 16000
  hop_size = 320
  min_value = np.log(1e-5)  # Matches MelGAN training.
  max_value = 4.0  # Largest value for most examples.
  pad_value = np.log(1e-5)  # Matches MelGAN training.
  # 16 extra frames are needed to avoid numerical errors during the mel bin
  # matmul.
  # The numerical errors are small, but enough to produce audible pops when
  # decoded by MelGan.
  additional_frames_for_encoding = 16

  def __init__(self, decode_dither_amount: float = 0.0):
    self._frame_length = 640
    self._fft_size = 1024
    self._lo_hz = 0.0
    self._decode_dither_amount = decode_dither_amount

  def encode(self, audio):
    """Compute features from audio.

    Args:
      audio: Shape [batch, n_samples].

    Returns:
      mel_spectrograms: Shape [batch, n_samples // hop_size, n_dims].
    """
    if tf.shape(audio)[0] == 0:
      # The STFT code doesn't always handle 0-length inputs correctly.
      # If we know the output is 0-length, just use this hard coded response.
      return tf.zeros((0, self.n_dims), dtype=tf.float32)
    return Audio2Mel(
        sample_rate=self.sample_rate,
        hop_length=self.hop_size,
        win_length=self._frame_length,
        n_fft=self._fft_size,
        n_mel_channels=self.n_dims,
        drop_dc=True,
        mel_fmin=self._lo_hz,
        mel_fmax=int(self.sample_rate // 2))(audio)

  def decode(self, features):
    """Decodes features to audio.

    Args:
      features: Mel spectrograms, shape [batch, n_frames, n_dims].

    Returns:
      audio: Shape [batch, n_frames * hop_size]
    """
    model = _load_model_from_cache('melgan')

    if self._decode_dither_amount > 0:
      features += (
          np.random.normal(size=features.shape) * self._decode_dither_amount)

    return model(features).numpy()  # pylint: disable=not-callable
