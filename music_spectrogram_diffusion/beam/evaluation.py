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

r"""Evaluate model in a beam job.

Note that this job is likely infeasible without access to an execution cluster
with accelerators, but it may still be useful as an example.
"""

from collections.abc import Sequence
import functools
import os
import re
import time
from typing import Any, Iterable, Mapping, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import jax
from music_spectrogram_diffusion import inference
from music_spectrogram_diffusion import transcription_inference
import music_spectrogram_diffusion.tasks  # pylint: disable=unused-import
import note_seq
import numpy as np
import seqio
from t5x import gin_utils
import tensorflow as tf


_CHECKPOINT_DIR = flags.DEFINE_string(
    'checkpoint_dir', None,
    'Path to checkpoint directory with model weights and gin config.',
    required=True)

_GIN_BINDINGS = flags.DEFINE_multi_string(
    'gin_bindings', default=[], help='Individual gin bindings.')

_TASKS = flags.DEFINE_list(
    'tasks', None,
    'Regexes matching task(s) to do inference for.', required=True)

_EXCLUDED_TASKS = flags.DEFINE_list(
    'excluded_tasks', None, 'Regexes matching task(s) to skip.')

_SPLIT = flags.DEFINE_string(
    'split', None, 'Split to use for inference.', required=True)

_USE_CACHED = flags.DEFINE_boolean(
    'use_cached', False, 'Whether to use cached dataset.')

_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Output path.')

_ALWAYS_MASK_CONTEXT = flags.DEFINE_boolean(
    'always_mask_context', False,
    'Whether context from the previous chunk should always be masked.')

_PREDICTION_SOURCE = flags.DEFINE_enum(
    'prediction_source', 'model', ['model', 'gt_raw', 'gt_encoded'],
    'What to use as the prediction source.')

FULL_RAW_AUDIO_PER_TASK = 3


# Trying to setup/teardown the model with each shard leads to TPU OOMs, likely
# because something remains allocated on the TPU. Using a per-process
# cached model seems to avoid this problem.


@functools.lru_cache()
def get_synth_model(checkpoint_dir: str, gin_config: str):
  return inference.InferenceModel(checkpoint_dir, gin_config)


@functools.lru_cache()
def get_transcription_model(checkpoint_dir: str):
  return transcription_inference.TranscriptionInferenceModel(checkpoint_dir)


class ProcessDataset(beam.DoFn):
  """Process seqio pipeline for one shard of the dataset."""

  def __init__(self, task: seqio.Task, num_shards: int,
               checkpoint_dir: str, gin_config: str, split: str,
               use_cached: bool):
    self._task = task
    self._num_shards = num_shards
    self._checkpoint_dir = checkpoint_dir
    self._gin_config = gin_config
    self._split = split
    self._use_cached = use_cached

  def setup(self):
    self._synth_model = get_synth_model(
        self._checkpoint_dir, self._gin_config)

  def teardown(self):
    # Free up TPU resources if this instance is being shut down.
    self._synth_model = None

  def process(
      self, shard_id: int
  ) -> Iterable[Tuple[Tuple[str, int, int], Tuple[int, Mapping[str, Any]]]]:
    logging.info('%s:%s: Processing shard %d of %d',
                 self._task.name, self._split, shard_id, self._num_shards)
    ds = self._task.get_dataset(
        sequence_length=self._synth_model.sequence_length,
        split=self._split,
        use_cached=self._use_cached,
        shuffle=False,
        shard_info=seqio.ShardInfo(index=shard_id, num_shards=self._num_shards))

    for i, ex in enumerate(ds.as_numpy_iterator()):
      beam.metrics.Metrics.counter(f'{self._task.name}_{self._split}',
                                   'processed_example').inc()
      song_key = (self._task.name, shard_id, ex['unique_id'][0])
      yield song_key, (i, ex)


class InferSong(beam.DoFn):
  """Infer a full song."""

  def __init__(self, task: seqio.Task, split: str,
               checkpoint_dir: str, gin_config: str,
               always_mask_context: bool, prediction_source: str):
    self._task = task
    self._split = split
    self._checkpoint_dir = checkpoint_dir
    self._gin_config = gin_config
    self._always_mask_context = always_mask_context
    self._prediction_source = prediction_source

  def setup(self):
    self._synth_model = get_synth_model(
        self._checkpoint_dir, self._gin_config)
    self._feature_converter = self._synth_model.model.FEATURE_CONVERTER_CLS(
        pack=False)

  def teardown(self):
    # Free up TPU resources if this instance is being shut down.
    self._feature_converter = None
    self._synth_model = None

  def process(self, id_examples):
    id_, examples = id_examples

    ac = self._synth_model.audio_codec

    # Initialize previous prediction to zeros.
    # We'll zero out the mask for the first prediction, so the value here
    # doesn't matter.
    pred_encoded = np.zeros(
        [1, self._synth_model.targets_context_length or 0, ac.n_dims],
        np.float32)

    # Variables for accumulating the full song prediction.
    full_pred_encoded = np.zeros([1, 0, ac.n_dims], np.float32)
    full_gt_encoded = np.zeros([1, 0, ac.n_dims], np.float32)
    full_gt_raw_audio = np.zeros([1, 0], np.float32)

    # Also keep the ground truth NoteSequence; it should be present in every
    # chunk.
    sequence = None

    prediction_seconds = []

    for i, (idx, example) in enumerate(sorted(examples)):
      del idx
      sequence = note_seq.NoteSequence.FromString(example['sequence'])

      ds = tf.data.Dataset.from_tensors(example)
      ds = self._feature_converter(ds, self._synth_model.sequence_length)
      ds = ds.batch(1)
      batches = list(ds.as_numpy_iterator())
      assert len(batches) == 1
      batch = batches[0]

      # Handle context inputs if the model expects them.
      if 'encoder_continuous_inputs' in batch:
        # TODO(fjord): This logic will need to be reworked if we start using a
        # context that is a different length than the predictions.
        batch['encoder_continuous_inputs'] = pred_encoded[:1]
        if i == 0 or self._always_mask_context:
          # The first chunk has no previous context.
          batch['encoder_continuous_mask'] = np.zeros_like(
              batch['encoder_continuous_mask'])
        else:
          # The full song pipeline does not feed in a context feature, so the
          # mask will be all 0s after the feature converter. Because we know
          # we're feeding in a full context chunk from the previous prediction,
          # set it to all 1s.
          batch['encoder_continuous_mask'] = np.ones_like(
              batch['encoder_continuous_mask'])

      tick = time.time()
      if self._prediction_source == 'model':
        pred_encoded, _ = jax.tree_map(
            np.asarray, self._synth_model.predict(batch))
      elif self._prediction_source in ('gt_raw', 'gt_encoded'):
        pred_encoded = batch['decoder_target_tokens']
      else:
        raise ValueError(
            f'Unknown prediction source: {self._prediction_source}')

      if i != 0:
        # Don't record time for the first segment because it may include
        # checkpoint loading, model compilation, etc.
        prediction_seconds.append(time.time() - tick)

      full_pred_encoded = np.concatenate(
          [full_pred_encoded, pred_encoded[:1]], axis=1)
      full_gt_encoded = np.concatenate(
          [full_gt_encoded, batch['decoder_target_tokens'][:1]], axis=1)
      full_gt_raw_audio = np.concatenate(
          [full_gt_raw_audio, np.expand_dims(example['raw_targets'], axis=0)],
          axis=1)
      logging.info('song id %s generated segment %d', id_, i)
      beam.metrics.Metrics.counter(f'{self._task.name}_{self._split}',
                                   'inferred_segment').inc()

    # Decode ground truth first so that any model caching happens here.
    full_gt_audio = ac.decode(full_gt_encoded)

    # Now that the model is definitely cached, time prediction decoding.
    tick = time.time()
    if self._prediction_source == 'gt_raw':
      full_pred_audio = full_gt_raw_audio
    else:
      full_pred_audio = ac.decode(full_pred_encoded)
    audio_decode_seconds = time.time() - tick

    assert len(prediction_seconds) == len(examples) - 1
    prediction_seconds_per_chunk = np.mean(prediction_seconds)
    audio_decode_seconds_per_chunk = audio_decode_seconds / len(examples)
    seconds_per_chunk = (
        self._synth_model.targets_length * (ac.hop_size / ac.sample_rate))
    prediction_seconds_per_audio_second = (
        prediction_seconds_per_chunk / seconds_per_chunk)
    audio_decode_seconds_per_audio_second = (
        audio_decode_seconds_per_chunk / seconds_per_chunk)

    assert sequence is not None

    beam.metrics.Metrics.counter(f'{self._task.name}_{self._split}',
                                 'inferred_song').inc()
    yield {
        'id': id_,
        'sequence': sequence,
        'full_gt_encoded': full_gt_encoded[0],
        'full_gt_audio': full_gt_audio[0],
        'full_gt_raw_audio': full_gt_raw_audio[0],
        'full_pred_audio': full_pred_audio[0],
        'full_pred_encoded': full_pred_encoded[0],
        'model_timing': {
            'prediction_seconds_per_chunk':
                prediction_seconds_per_chunk,
            'predictions_seconds_per_audio_second':
                prediction_seconds_per_audio_second,
            'audio_decode_seconds_per_chunk':
                audio_decode_seconds_per_chunk,
            'audio_decode_seconds_per_audio_second':
                audio_decode_seconds_per_audio_second,
        },
    }


class SerializeSong(beam.DoFn):
  """Serialize a full song."""

  def __init__(self, task: seqio.Task, split: str, checkpoint_dir: str,
               gin_config: str, output_dir: str):
    self._task = task
    self._split = split
    self._checkpoint_dir = checkpoint_dir
    self._gin_config = gin_config
    self._output_dir = output_dir

  def setup(self):
    self._synth_model = get_synth_model(
        self._checkpoint_dir, self._gin_config)

  def teardown(self):
    # Free up TPU resources if this instance is being shut down.
    self._synth_model = None

  def process(self, ex):
    """Write outputs as individual files for easy access."""
    tf.io.gfile.makedirs(self._output_dir)

    prefix = '_'.join(str(x) for x in ex['id'])

    pred_wav_data = note_seq.audio_io.samples_to_wav_data(
        ex['full_pred_audio'],
        sample_rate=self._synth_model.audio_codec.sample_rate)
    tf.io.gfile.GFile(
        os.path.join(self._output_dir, prefix + '_pred.wav'), 'wb').write(
            pred_wav_data)

    gt_wav_data = note_seq.audio_io.samples_to_wav_data(
        ex['full_gt_audio'],
        sample_rate=self._synth_model.audio_codec.sample_rate)
    tf.io.gfile.GFile(
        os.path.join(self._output_dir, prefix + '_gt.wav'), 'wb').write(
            gt_wav_data)

    gt_raw_wav_data = note_seq.audio_io.samples_to_wav_data(
        ex['full_gt_raw_audio'],
        sample_rate=self._synth_model.audio_codec.sample_rate)
    tf.io.gfile.GFile(
        os.path.join(self._output_dir, prefix + '_gt_raw.wav'), 'wb').write(
            gt_raw_wav_data)

    np.save(tf.io.gfile.GFile(
        os.path.join(self._output_dir, prefix + '_gt_encoded.npy'), 'wb'),
            ex['full_gt_encoded'])

    np.save(tf.io.gfile.GFile(
        os.path.join(self._output_dir, prefix + '_pred_encoded.npy'), 'wb'),
            ex['full_pred_encoded'])

    beam.metrics.Metrics.counter(
        f'{self._task.name}_{self._split}', 'serialized').inc()


class PostProcessForMetrics(beam.DoFn):
  """Postprocess song for metrics."""

  def __init__(self, task: seqio.Task, split: str, checkpoint_dir: str,
               gin_config: str, prediction_source: str):
    self._task = task
    self._split = split
    self._checkpoint_dir = checkpoint_dir
    self._gin_config = gin_config
    self._prediction_source = prediction_source

  def setup(self):
    self._synth_model = get_synth_model(
        self._checkpoint_dir, self._gin_config)

  def teardown(self):
    # Free up TPU resources if this instance is being shut down.
    self._synth_model = None

  def process(
      self, ex, include_raw_audio_ids: Iterable[Tuple[Any]]
  ) -> Iterable[Tuple[str, Tuple[Any, Any]]]:
    """Run outputs through task postprocessor to be ready for metrics."""
    assert self._task.postprocessor

    def gen_target_prediction(suffix: str,
                              raw_start: Optional[int],
                              raw_end: Optional[int],
                              enc_start: Optional[int],
                              enc_end: Optional[int],
                              include_raw_audio: bool):
      if raw_start is not None:
        start_time = raw_start / self._synth_model.audio_codec.sample_rate
        start_time = min(start_time, ex['sequence'].total_time)
        end_time = raw_end / self._synth_model.audio_codec.sample_rate
        end_time = min(end_time, ex['sequence'].total_time)
        if start_time == end_time:
          # There should be no notes in this segment.
          sequence = note_seq.NoteSequence(ticks_per_quarter=220)
        else:
          sequence = note_seq.extract_subsequence(
              ex['sequence'], start_time, end_time)
      else:
        sequence = ex['sequence']

      example = {
          'raw_targets': ex['full_gt_raw_audio'][raw_start:raw_end],
          'unique_id': ex['id'],
          'sequence': sequence,
          'model_timing': ex['model_timing'],
      }

      include_raw_audio = include_raw_audio and ex['id'] in include_raw_audio_ids

      target = self._task.postprocessor(
          ex['full_gt_encoded'][enc_start:enc_end], example, is_target=True,
          include_raw_audio=include_raw_audio)
      prediction = self._task.postprocessor(
          ex['full_pred_encoded'][enc_start:enc_end], example, is_target=False,
          include_raw_audio=include_raw_audio,
          use_raw_targets_as_prediction=self._prediction_source == 'gt_raw')

      task_metrics_name = f'{self._task.name}{suffix}'
      beam.metrics.Metrics.counter(f'{task_metrics_name}_{self._split}',
                                   'postprocessed_for_metrics').inc()

      return task_metrics_name, (target, prediction)

    def get_idxs_for_minute(minute):
      ac = self._synth_model.audio_codec
      chunk_frames = self._synth_model.targets_length
      chunk_idx = int(
          ((ac.sample_rate * minute * 60) / ac.hop_size) / chunk_frames)

      frame_start = chunk_idx * chunk_frames
      frame_end = (chunk_idx + 1) * chunk_frames

      return {
          'raw_start': frame_start * ac.hop_size,
          'raw_end': frame_end * ac.hop_size,
          'enc_start': frame_start,
          'enc_end': frame_end,
      }

    # First do the full example, with a max of 10 minutes of audio.
    idxs_10minutes = get_idxs_for_minute(10)
    yield gen_target_prediction(suffix='',
                                raw_start=None,
                                raw_end=idxs_10minutes['raw_end'],
                                enc_start=None,
                                enc_end=idxs_10minutes['enc_end'],
                                include_raw_audio=True)

    # Do up to 10 minutes' worth of chunks.
    for minute in range(11):
      idxs_minute = get_idxs_for_minute(minute)

      # Check against the ground truth encoded because it won't have any
      # padding.
      if len(ex['full_gt_encoded']) < idxs_minute['enc_end']:
        continue

      yield gen_target_prediction(
          suffix=f'_{minute}minute', **idxs_minute, include_raw_audio=False)


# This is a separate "postprocessing" step that we basically never want to do
# during normal training, as running inference on the transcription model is
# extremely slow.
class TranscribeAudio(beam.DoFn):
  """Transcribe ground truth and synthesized audio."""

  _TRANSCRIPTION_CKPT_PATH = 'gs://mt3/checkpoints/mt3/'

  def setup(self):
    self._model = get_transcription_model(self._TRANSCRIPTION_CKPT_PATH)

  def teardown(self):
    self._model = None

  def process(self, kv):
    key, (target, prediction) = kv
    prediction['transcribed_audio'] = self._model(prediction['audio'])
    target['transcribed_audio'] = self._model(target['raw_targets'])
    beam.metrics.Metrics.counter(key, 'transcribed_audio').inc()
    yield key, (target, prediction)


def drop_raw_audio(kv):
  key, (target, pred) = kv
  if not target['include_raw_audio']:
    del target['raw_targets']
  if not pred['include_raw_audio']:
    del pred['audio']
  return key, (target, pred)


class ComputeMetrics(beam.DoFn):
  """Compute task metrics."""

  def __init__(self, task: seqio.Task, split: str, checkpoint_dir: str,
               gin_config: str, output_dir: str):
    self._task = task
    self._split = split
    self._checkpoint_dir = checkpoint_dir
    self._gin_config = gin_config
    self._output_dir = output_dir

  def setup(self):
    self._synth_model = get_synth_model(
        self._checkpoint_dir, self._gin_config)

  def teardown(self):
    # Free up TPU resources if this instance is being shut down.
    self._synth_model = None

  def process(
      self, task_metrics_name_examples: Tuple[str, Iterable[Tuple[Any, Any]]]):
    # task_metrics_name should be the same as task.name, but with an optional
    # suffix.
    task_metrics_name, examples = task_metrics_name_examples
    assert task_metrics_name.startswith(self._task.name)

    # Put examples with IDs first, in sorted order.
    examples = sorted(
        [ex for ex in examples if 'unique_id' in ex[0]],
        key=lambda ex: ex[0]['unique_id']) + [
            ex for ex in examples if 'unique_id' not in ex[0]
        ]

    targets = [x[0] for x in examples]
    predictions = [x[1] for x in examples]

    metrics = {}
    for metric_fn in self._task.predict_metric_fns:
      for k, v in metric_fn(targets, predictions).items():
        logging.info('Adding %s metrics for Task %s', k, task_metrics_name)
        beam.metrics.Metrics.counter(f'{task_metrics_name}_{self._split}',
                                     'computed_metric').inc()
        if k in metrics:
          raise ValueError(
              f"Duplicate metric key '{k}' in Task '{task_metrics_name}'.")
        metrics[k] = (seqio.metrics.Scalar(v)
                      if not isinstance(v, seqio.metrics.MetricValue) else v)

    for logger_cls in (seqio.PyLoggingLogger, seqio.TensorBoardLogger):
      logger = logger_cls(self._output_dir)
      logger(task_name=task_metrics_name,
             step=self._synth_model.step,
             metrics=metrics,
             dataset=None,
             inferences=None,
             targets=None)

    beam.metrics.Metrics.counter(f'{task_metrics_name}_{self._split}',
                                 'full_metrics_logged').inc()


# Add some convenience imports needed for most gin overrides.
# pylint: disable=line-too-long
COMMON_GIN_BINDINGS = [
    'from __gin__ import dynamic_registration',
    'from music_spectrogram_diffusion.models.diffusion import diffusion_utils',
]
# pylint: enable=line-too-long


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if tf.io.gfile.exists(_OUTPUT_PATH.value):
    raise ValueError(f'Output directory {_OUTPUT_PATH.value} already exists.')
  tf.io.gfile.makedirs(_OUTPUT_PATH.value)

  gin_path = os.path.join(_CHECKPOINT_DIR.value, '..', 'config.gin')

  # Parse the gin config along with any added flags here.
  # Also includes logic for rewriting old import paths, etc.
  # Then, save the resulting combined config string to be passed along to the
  # different beam stages to be used during model restores.
  logging.info('Parsing gin config at %s', gin_path)
  parsed_gin_config = inference.parse_training_gin_file(
      gin_path, COMMON_GIN_BINDINGS + _GIN_BINDINGS.value)

  config_txt = f"""checkpoint_dir={_CHECKPOINT_DIR.value}
always_mask_context={_ALWAYS_MASK_CONTEXT.value}
prediction_source={_PREDICTION_SOURCE.value}

Parsed gin:
{parsed_gin_config}
"""

  tf.io.gfile.GFile(
      os.path.join(_OUTPUT_PATH.value, 'config.txt'), 'w').write(
          config_txt)

  included_regex = re.compile(r'(%s\Z)' % r'\Z|'.join(_TASKS.value))
  # Excludes only empty names by default.
  excluded_regex = re.compile(
      r'(%s\Z)' % r'\Z|'.join(_EXCLUDED_TASKS.value or []))
  task_names = [
      t for t in seqio.TaskRegistry.names()
      if included_regex.match(t) and not excluded_regex.match(t)]
  if not task_names:
    logging.warning("No tasks have been selected from the task registry. "
                    "Please make sure that the tasks you want cached exist in "
                    "the task registry and haven't been excluded by the "
                    "--excluded_tasks flag.")

  tasks = []
  for task_name in task_names:
    logging.info('Adding task %s', task_name)
    tasks.append(seqio.TaskRegistry.get(task_name))

  with beam.Pipeline() as p:
    for task in tasks:
      if _USE_CACHED.value:
        # TODO(fjord): Should this be a more official seqio API?
        # There's currently no standard way to get the number of shards in a
        # cached dataset.
        num_shards = len(
            task._get_cached_source(  # pylint: disable=protected-access
                _SPLIT.value).list_shards(_SPLIT.value))
      else:
        num_shards = len(task.source.list_shards(_SPLIT.value))
      logging.info('Using %d preprocessing shards for task %s:%s',
                   num_shards, task.name, _SPLIT.value)
      task_output_dir = os.path.join(
          _OUTPUT_PATH.value, seqio.get_task_dir_from_name(task.name))

      inferred_songs = (
          p
          | f'{task.name}_create_shards' >> beam.Create(list(range(num_shards)))
          | f'{task.name}_process_dataset' >> beam.ParDo(
              ProcessDataset(
                  task=task,
                  num_shards=num_shards,
                  checkpoint_dir=_CHECKPOINT_DIR.value,
                  gin_config=parsed_gin_config,
                  split=_SPLIT.value,
                  use_cached=_USE_CACHED.value))
          | f'{task.name}_group_by_unique_id' >> beam.GroupByKey()
          | f'{task.name}_reshuffle' >> beam.Reshuffle()
          | f'{task.name}_infer' >> beam.ParDo(
              InferSong(
                  task=task,
                  split=_SPLIT.value,
                  checkpoint_dir=_CHECKPOINT_DIR.value,
                  gin_config=parsed_gin_config,
                  always_mask_context=_ALWAYS_MASK_CONTEXT.value,
                  prediction_source=_PREDICTION_SOURCE.value)
          ).with_resource_hints(
              # Use a pool that is separate from the transcription
              # pool to avoid issues with multiple models on the same TPU.
              tags='inference',
              flume_accelerator_config=(
                  # TODO(fjord): specify accelerator resources here.
                  ))
          | f'{task.name}_post_infer_reshuffle' >> beam.Reshuffle())

      # Serialize
      _ = (
          inferred_songs
          | f'{task.name}_serialize' >> beam.ParDo(
              SerializeSong(
                  task=task,
                  split=_SPLIT.value,
                  checkpoint_dir=_CHECKPOINT_DIR.value,
                  gin_config=parsed_gin_config,
                  output_dir=task_output_dir)))

      def get_first_n_ids(ids):
        # Entries may be id tuples or lists of id tuples. Flatten first.
        all_ids = []
        for id_ in ids:
          if isinstance(id_, list):
            all_ids.extend(id_)
          elif isinstance(id_, tuple):
            all_ids.append(id_)
          else:
            raise ValueError(f'Unexpected type: {type(id_)}: {id_}')
        return sorted(all_ids)[:FULL_RAW_AUDIO_PER_TASK]

      # Postprocess and Metrics
      first_n_ids = (
          inferred_songs
          | f'{task.name}_get_ids' >> beam.Map(lambda ex: ex['id'])
          | f'{task.name}_first_n' >> beam.CombineGlobally(get_first_n_ids))
      _ = (
          inferred_songs
          | f'{task.name}_postprocess' >> beam.ParDo(
              PostProcessForMetrics(
                  task=task,
                  split=_SPLIT.value,
                  checkpoint_dir=_CHECKPOINT_DIR.value,
                  gin_config=parsed_gin_config,
                  prediction_source=_PREDICTION_SOURCE.value),
              include_raw_audio_ids=beam.pvalue.AsSingleton(first_n_ids))
          | f'{task.name}_transcribe' >> beam.ParDo(
              TranscribeAudio()
          ).with_resource_hints(
              # Use a pool that is separate from the inference
              # pool to avoid issues with multiple models on the same TPU.
              tags='transcription',
              flume_accelerator_config=(
                  # TODO(fjord): specify accelerator resources here.
                  ))
          | f'{task.name}_drop_raw_audio' >> beam.Map(drop_raw_audio)
          | f'{task.name}_group_by_task_name' >> beam.GroupByKey()
          | f'{task.name}_metrics' >> beam.ParDo(
              ComputeMetrics(
                  task=task,
                  split=_SPLIT.value,
                  checkpoint_dir=_CHECKPOINT_DIR.value,
                  gin_config=parsed_gin_config,
                  output_dir=_OUTPUT_PATH.value)
              ).with_resource_hints(worker_ram='128G'))


if __name__ == '__main__':
  gin_utils.run(main)
