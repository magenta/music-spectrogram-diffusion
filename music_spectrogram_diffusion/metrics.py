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

"""Synthesis metrics."""

import collections
import io

import clu.metrics
import clu.values
import flax
import imageio as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mt3 import metrics as transcription_metrics
from music_spectrogram_diffusion import audio_codecs
import numpy as np
from scipy import linalg
import seqio
from t5x import metrics as metrics_lib

MetricsMap = metrics_lib.MetricsMap


# -------------------------- Model (Train) Metrics -----------------------------
def compute_base_metrics(
    loss_weights: jnp.ndarray,
    loss: jnp.ndarray,
) -> MetricsMap:
  """Compute summary metrics.

  Args:
   loss_weights: Binary array of shape [batch, time, 1] for weighting the loss.
     Is zero anywhere padding is applied.
   loss: loss (float)

  Returns:
    Dict of metrics.
  """
  num_examples = loss_weights.shape[0]
  num_frames = jnp.sum(loss_weights)
  num_devices = jax.device_count()
  assert num_devices, 'JAX is reporting no devices, but it should.'
  metrics_dict = {
      'loss':
          metrics_lib.AveragePerStep(total=loss),
      'loss_per_target_frame':
          clu.metrics.Average(total=loss, count=num_frames),
      'target_frames':
          clu.metrics.Average(total=num_frames, count=num_examples),
      'timing/seqs_per_second':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples),
      'timing/steps_per_second':
          metrics_lib.StepsPerTime.from_model_output(),
      'timing/seconds':
          metrics_lib.Time(),
      'timing/seqs':
          metrics_lib.Sum(num_examples),
      'timing/seqs_per_second_per_core':
          metrics_lib.TimeRate.from_model_output(numerator=num_examples /
                                                 num_devices),
      'timing/target_frame_per_second':
          metrics_lib.TimeRate.from_model_output(numerator=num_frames),
      'timing/target_frame_per_second_per_core':
          metrics_lib.TimeRate.from_model_output(numerator=num_frames /
                                                 num_devices),
  }
  return metrics_dict


@flax.struct.dataclass
class ImageSummary(clu.metrics.LastValue):

  def compute_value(self):
    """Input values scaled to [0, 1]."""
    return clu.values.Image(self.value[:, :, :, None])


# -------------------------- Task Metrics --------------------------------------
def count_examples(targets, predictions):
  """Simple metric for counting the number of examples."""
  assert len(targets) == len(predictions)
  return {'examples_count': seqio.metrics.Scalar(len(targets))}


def model_timing(targets, predictions):
  del targets
  timings = collections.defaultdict(list)
  for prediction in predictions:
    if 'model_timing' in prediction:
      for k, v in prediction['model_timing'].items():
        timings[k].append(v)
  return {f'model_timing/{k}': seqio.metrics.Scalar(np.mean(v))
          for k, v in timings.items()}


def _pad_list_of_arrays(arrays, pad_value):
  """Pads and concatentates a list of arrays into a single array."""
  lengths = [len(t) for t in arrays]
  max_length = max(lengths)

  padded = []
  for t, l in zip(arrays, lengths):
    t = np.array(t)
    padding = [[0, int(max_length - l)]]
    if len(t.shape) == 2:
      padding += [[0, 0]]
    padded.append(np.pad(t, padding, constant_values=pad_value))
  return np.array(padded, dtype=np.float32)


# The below functions related to mpl figure / spectrogram summaries are largely
# copied from: //third_party/py/ddsp/training/summaries.py


def _fig_image(fig):
  """Returns an image summary from a matplotlib figure."""
  buffer = io.BytesIO()
  fig.savefig(buffer, format='png', bbox_inches='tight')
  img = iio.imread(buffer.getvalue(), format='png')
  plt.close(fig)
  return img


def _plt_spec(spec, ax, title, vmin=0, vmax=1):
  """Helper function to plot a spectrogram to an axis."""
  spec = np.rot90(spec)
  ax.matshow(spec, vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.magma)
  ax.set_title(title)
  ax.set_xticks([])
  ax.set_yticks([])


def _spectrogram_summaries(targets, predictions):
  """Spectrogram summaries for a batch of audio targets and predictions."""
  assert len(targets) == len(predictions)
  imgs = []

  melgan = audio_codecs.MelGAN()
  targets = melgan.to_images(melgan.encode(targets))
  predictions = melgan.to_images(melgan.encode(predictions))

  for i in range(len(targets)):
    # Manually specify exact size of fig for tensorboard
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    _plt_spec(targets[i], axs[0], 'original')
    _plt_spec(predictions[i], axs[1], 'synthesized')

    # Format and save plot to image
    imgs.append(_fig_image(fig))

  return seqio.metrics.Image(np.stack(imgs))


def image_metric_fn(targets, predictions, audio_codec, context_keys=()):
  """Image summaries of network outputs for InferEval.

  These metric_fn's come straight from the task (no feature converter), so
  don't have padding, and need manual padding to make an array for image
  summaries.
  Args:
    targets: List of arrays
    predictions: List of arrays
    audio_codec: AudioCodec used to scale network outputs.
    context_keys: Keys for audio context.

  Returns:
    Dict of image summaries from concat/padded list of arrays.
  """
  # Filter any examples where the raw audio was dropped by the postprocessor.
  targets = [x for x in targets if 'raw_targets' in x]
  predictions = [x for x in predictions if 'audio' in x]
  assert len(targets) == len(predictions)
  if len(targets) == 0:  # pylint: disable=g-explicit-length-test
    return {}

  audio_targets = _pad_list_of_arrays(
      [target['raw_targets'] for target in targets], pad_value=0.0)
  audio_predictions = _pad_list_of_arrays(
      [pred['audio'] for pred in predictions], pad_value=0.0)

  output = {
      'spectrogram': _spectrogram_summaries(audio_targets, audio_predictions)
  }

  predictions = _pad_list_of_arrays(
      [pred['predictions'] for pred in predictions],
      pad_value=audio_codec.pad_value)
  output['predictions'] = seqio.metrics.Image(
      audio_codec.to_images(predictions)[..., None])

  # Collect targets.
  target_types = collections.defaultdict(list)
  for target in targets:
    target_types['targets'].append(target['targets'])
    if 'targets_context' in target:
      target_types['targets_context'].append(target['targets_context'])

  for k, v in target_types.items():
    ac = audio_codec.context_codec if k in context_keys else audio_codec
    v = _pad_list_of_arrays(v, pad_value=ac.pad_value)
    output[k] = seqio.metrics.Image(ac.to_images(v)[..., None])

  return output

# TensorBoard has a 10mb limit for audio metrics. Ensures we stay
# under that, while still allowing a little over 1 minute at 16khz.
MAX_AUDIO_METRIC_SAMPLES = 1_000_000


def audio_metric_fn(targets, predictions, audio_codec,
                    context_keys=(), max_outputs=3):
  """Audio summaries from network outputs for InferEval.

  These metric_fn's come straight from the task (no feature converter), so
  don't have padding, and need manual padding to make an array for image
  summaries.
  Args:
    targets: List of arrays
    predictions: List of arrays
    audio_codec: AudioCodec used to convert network outputs to audio.
    context_keys: Keys for audio context.
    max_outputs: Max number of outputs to create summaries.

  Returns:
    Dict of audio metrics synthesized from network outputs.
  """
  sr = audio_codec.sample_rate

  # Filter any examples where the raw audio was dropped by the postprocessor.
  targets = [x for x in targets if 'raw_targets' in x]
  predictions = [x for x in predictions if 'audio' in x]
  assert len(targets) == len(predictions)
  if len(targets) == 0:  # pylint: disable=g-explicit-length-test
    return {}

  audio_predictions = _pad_list_of_arrays(
      [pred['audio'] for pred in predictions], pad_value=0.0)
  output = {
      'audio_predictions': seqio.metrics.Audio(
          audio_predictions[:, :MAX_AUDIO_METRIC_SAMPLES, None], sr,
          max_outputs)
  }

  # Collect targets.
  target_types = collections.defaultdict(list)
  for target in targets:
    target_types['targets'].append(target['targets'])
    target_types['raw_targets'].append(target['raw_targets'])
    if 'targets_context' in target:
      target_types['targets_context'].append(target['targets_context'])
      target_types['raw_targets_context'].append(target['raw_targets_context'])

  for k, v in target_types.items():
    # Use context codec for decoding context audio.
    ac = audio_codec.context_codec if k in context_keys else audio_codec

    if k.startswith('raw_'):
      v = _pad_list_of_arrays(v, pad_value=0.0)[:max_outputs]
      output[f'audio_{k}'] = seqio.metrics.Audio(
          v[:, :MAX_AUDIO_METRIC_SAMPLES, None], sr, max_outputs)
    else:
      v = _pad_list_of_arrays(v, pad_value=ac.pad_value)[:max_outputs]
      output[f'audio_resynth_{k}'] = seqio.metrics.Audio(
          ac.decode(v)[:, :MAX_AUDIO_METRIC_SAMPLES, None], sr, max_outputs)

  return output


def _embedding_distance(
    embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> float:
  """Compute distance between two audio clips under all model embeddings."""
  num_frames = min(len(embeddings_1), len(embeddings_2))
  diff = embeddings_1[:num_frames] - embeddings_2[:num_frames]
  return np.mean(np.linalg.norm(diff, axis=1))


# copied from //audio/ears/vivaldi/eval/fad/fad_utils.py
def _stable_trace_sqrt_product(sigma_test, sigma_train, eps=1e-7):
  """Avoids some problems when computing the srqt of product of sigmas.

  Based on Dougal J. Sutherland's contribution here:
  https://github.com/bioinf-jku/TTUR/blob/master/fid.py

  Args:
    sigma_test: Test covariance matirx.
    sigma_train: Train covariance matirx.
    eps: Small number; used to avoid singular product.

  Returns:
    The Trace of the squre root of the product of the passed convariance
    matrices.

  Raises:
    ValueError: If the sqrt of the product of the sigmas contains complex
        numbers with large imaginary parts.
  """
  # product might be almost singular
  sqrt_product, _ = linalg.sqrtm(sigma_test.dot(sigma_train), disp=False)
  if not np.isfinite(sqrt_product).all():
    # add eps to the diagonal to avoid a singular product.
    offset = np.eye(sigma_test.shape[0]) * eps
    sqrt_product = linalg.sqrtm((sigma_test + offset).dot(sigma_train + offset))

  # Might have a slight imaginary component.
  # Note: increased tolerance from 1e-3 to 3e-3 because comparing raw audio
  # to itself resulted in a max imag component of ~.002.
  if not np.allclose(np.diagonal(sqrt_product).imag, 0, atol=3e-3):
    m = np.max(np.abs(sqrt_product.imag))
    raise ValueError(f'sqrt_product contains large complex numbers: {m}')
  sqrt_product = sqrt_product.real

  return np.trace(sqrt_product)


def _frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
  """Compute Fréchet distance between two multivariate Gaussians."""
  assert mu_1.shape == mu_2.shape
  assert sigma_1.shape == sigma_2.shape
  mu_diff = mu_1 - mu_2
  mu_dist = mu_diff.dot(mu_diff)
  if len(sigma_1.shape) == 2:
    # Full covariance matrix.
    trace_sqrt_product = _stable_trace_sqrt_product(sigma_1, sigma_2)
    return (mu_dist + np.trace(sigma_1) + np.trace(sigma_2) -
            2 * trace_sqrt_product)
  else:
    # Diagonal covariance.
    return (mu_dist + np.sum(sigma_1) + np.sum(sigma_2) -
            2 * np.sum(np.sqrt(sigma_1 * sigma_2)))


class StreamingMultivariateGaussian(object):
  """Streaming mean and covariance for multivariate Gaussian."""

  # If dimension is greater than this, use diagonal covariance.
  _MAX_FULL_COVARIANCE_DIM = 1024

  def __init__(self):
    self.n = 0
    self.mu = None
    self._sigma_accum = None

  def update(self, x):
    """Update mean and covariance with new data points."""
    n, d = x.shape
    if self.n == 0:
      self.n = n
      self.mu = np.mean(x, axis=0)
      x_res = x - self.mu[np.newaxis, :]
      if d <= self._MAX_FULL_COVARIANCE_DIM:
        self._sigma_accum = np.dot(x_res.T, x_res)
      else:
        self._sigma_accum = np.sum(x_res * x_res, axis=0)
    else:
      x_res_pre = x - self.mu[np.newaxis, :]
      self.n += n
      self.mu += np.sum(x_res_pre, axis=0) / self.n
      x_res_post = x - self.mu[np.newaxis, :]
      if d <= self._MAX_FULL_COVARIANCE_DIM:
        self._sigma_accum += np.dot(x_res_pre.T, x_res_post)
      else:
        self._sigma_accum += np.sum(x_res_pre * x_res_post, axis=0)

  @property
  def sigma(self):
    return self._sigma_accum / self.n


def reconstruction_metric_fn(targets, predictions):
  """Compute mean audio reconstruction distance across examples."""
  scores = collections.defaultdict(list)

  # Maintain mean and covariance for each collection of embeddings.
  target_gaussians = collections.defaultdict(StreamingMultivariateGaussian)
  pred_gaussians = collections.defaultdict(StreamingMultivariateGaussian)

  for target, prediction in zip(targets, predictions):
    target_embeddings = target['embeddings']
    prediction_embeddings = prediction['embeddings']

    assert target_embeddings.keys() == prediction_embeddings.keys()

    for embedding_type in target_embeddings:
      dist = _embedding_distance(
          target_embeddings[embedding_type],
          prediction_embeddings[embedding_type])
      scores[embedding_type + '_distance'].append(dist)
      target_gaussians[embedding_type].update(
          target_embeddings[embedding_type])
      pred_gaussians[embedding_type].update(
          prediction_embeddings[embedding_type])

  result = {k: np.mean(v) for k, v in scores.items()}

  # Compute Fréchet audio distance for each embedding type.
  assert set(target_gaussians.keys()) == set(pred_gaussians.keys())
  for name in target_gaussians:
    mu_target = target_gaussians[name].mu
    sigma_target = target_gaussians[name].sigma
    mu_pred = pred_gaussians[name].mu
    sigma_pred = pred_gaussians[name].sigma
    result[name + '_fréchet'] = _frechet_distance(
        mu_target, sigma_target, mu_pred, sigma_pred)

  return result


def transcription_metric_fn(targets, predictions):
  """Compute transcription metrics (for evaluating synthesis)."""
  scores = collections.defaultdict(list)

  for target, prediction in zip(targets, predictions):
    if 'transcribed_audio' not in prediction:
      continue

    ns_target_gt = target['sequence']
    ns_target_transcribed = target['transcribed_audio']
    ns_pred_transcribed = prediction['transcribed_audio']

    for granularity_type in ['flat', 'midi_class', 'full']:
      for name, score in transcription_metrics.program_aware_note_scores(
          ns_target_gt, ns_pred_transcribed,
          granularity_type=granularity_type).items():
        scores['Model ' + name].append(score)
      # Also report transcription metrics for the ground truth *audio* as a
      # sort of ceiling; we shouldn't expect to do better than that.
      for name, score in transcription_metrics.program_aware_note_scores(
          ns_target_gt, ns_target_transcribed,
          granularity_type=granularity_type).items():
        scores['GT ' + name].append(score)

  return {k: np.mean(v) for k, v in scores.items()}
