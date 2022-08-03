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

"""Transcription task definitions."""

import dataclasses
import functools
from typing import Callable, Optional, Sequence

from absl import logging
from music_spectrogram_diffusion import audio_codecs
from music_spectrogram_diffusion import datasets
from music_spectrogram_diffusion import event_codec
from music_spectrogram_diffusion import metrics
from music_spectrogram_diffusion import postprocessors
from music_spectrogram_diffusion import preprocessors
from music_spectrogram_diffusion import run_length_encoding
from music_spectrogram_diffusion import vocabularies
import seqio
import t5
import tensorflow as tf

partial = functools.partial


# Split audio frame sequences into this length before the cache placeholder.
MAX_NUM_CACHED_FRAMES = 2000

seqio.add_global_cache_dirs([])


@dataclasses.dataclass
class NoteRepresentationConfig:
  """Configuration note representations."""
  onsets_only: bool
  include_ties: bool


def construct_task_name(
    task_prefix: str,
    dataset_name: str,
    audio_codec: audio_codecs.AudioCodec,
    vocab_config: vocabularies.VocabularyConfig,
    note_representation_config: NoteRepresentationConfig,
    task_suffix: Optional[str] = None
) -> str:
  """Construct task name from prefix, config, and optional suffix."""
  # Name the tasks.
  task_type = 'onsets' if note_representation_config.onsets_only else 'notes'
  if note_representation_config.include_ties:
    task_type += '_ties'

  fields = [task_prefix]
  fields.append(dataset_name)
  fields.append(task_type)
  if audio_codec.abbrev_str:
    fields.append(audio_codec.abbrev_str)
  if vocab_config.abbrev_str:
    fields.append(vocab_config.abbrev_str)
  if task_suffix:
    fields.append(task_suffix)
  return '_'.join(fields)


def pre_cache_processor_chain(
    audio_codec: audio_codecs.AudioCodec,
    codec: event_codec.Codec,
    tokenize_fn,  # TODO(iansimon): add type signature
    note_representation_config: NoteRepresentationConfig,
    split_sequences=True):
  """Return list of preprocessors to run before the cache placeholder."""
  precache_preprocessors = [
      partial(
          tokenize_fn,
          audio_codec=audio_codec,
          codec=codec,
          is_training_data=True,
          onsets_only=note_representation_config.onsets_only,
          include_ties=note_representation_config.include_ties),
      # Using transcription tokenization functions (Input: Audio, Target: Notes)
      # Rekey for Synthesis (Input:Notes, Target: Audio)
      partial(
          t5.data.preprocessors.rekey,
          key_map={
              'inputs': 'targets',
              'target_times': 'input_times',
              'targets': 'inputs',
              # Need to include identity mappings to not lose entries.
              'event_start_indices': 'event_start_indices',
              'event_end_indices': 'event_end_indices',
              'state_events': 'state_events',
              'state_event_indices': 'state_event_indices',
              'sequence': 'sequence',
          })
  ]
  if split_sequences:
    precache_preprocessors.append(
        partial(
            t5.data.preprocessors.split_tokens,
            max_tokens_per_segment=MAX_NUM_CACHED_FRAMES,
            feature_key='targets',
            additional_feature_keys=[
                'event_start_indices',
                'event_end_indices',
                'state_event_indices'
            ],
            passthrough_feature_keys=['inputs', 'state_events']))
  return precache_preprocessors


def split_full_song_processor_chain(
    audio_codec: audio_codecs.AudioCodec,
    feature_context_key: Optional[str] = None):
  """Return a list of preprocessors to split a full song into examples."""
  split_full_song_preprocessors = [
      preprocessors.add_unique_id,
      partial(
          preprocessors.split_full_song,
          feature_key='targets',
          audio_codec=audio_codec,
          additional_feature_keys=[
              'event_start_indices',
              'event_end_indices',
              'state_event_indices'
          ],
          passthrough_feature_keys=[
              'inputs', 'state_events', 'unique_id', 'sequence'
          ])
  ]
  if feature_context_key is not None:
    def add_empty_context(ex):
      ex[feature_context_key] = tf.zeros((0,) + ex['targets'].shape[1:],
                                         dtype=ex['targets'].dtype)
      return ex
    split_full_song_preprocessors.append(
        seqio.map_over_dataset(add_empty_context))
  return split_full_song_preprocessors


def note_representation_processor_chain(
    codec: event_codec.Codec,
    note_representation_config: NoteRepresentationConfig):

  tie_token = codec.encode_event(event_codec.Event('tie', 0))
  state_events_end_token = tie_token if note_representation_config.include_ties else None

  return [
      partial(
          run_length_encoding.extract_sequence_with_indices,
          state_events_end_token=state_events_end_token,
          feature_key='inputs'),
      partial(
          preprocessors.map_midi_programs,
          feature_key='inputs',
          codec=codec),
      run_length_encoding.run_length_encode_shifts_fn(
          codec,
          feature_key='inputs',
          state_change_event_types=['velocity', 'program']),
  ]


def construct_train_eval_tasks_and_mixture(
    task_prefix: str,
    dataset_config: datasets.DatasetConfig,
    audio_codec: audio_codecs.AudioCodec,
    vocab_config: vocabularies.VocabularyConfig,
    note_representation_config: NoteRepresentationConfig,
    output_features: seqio.preprocessors.OutputFeaturesType,
    gen_task_preprocessors: Callable[..., Sequence[Callable[...,
                                                            tf.data.Dataset]]],
    metric_fns: Sequence[seqio.MetricFnCallable]) -> None:
  """Construct train eval tasks and mixture for the given config."""
  # Name the tasks.
  train_task_name = construct_task_name(
      task_prefix=task_prefix,
      dataset_name=dataset_config.name,
      audio_codec=audio_codec,
      vocab_config=vocab_config,
      note_representation_config=note_representation_config,
      task_suffix='train')

  mixture_task_names = []

  # Add training task.
  seqio.TaskRegistry.add(
      train_task_name,
      source=seqio.TFExampleDataSource(
          split_to_filepattern={
              'train': dataset_config.paths[dataset_config.train_split],
              'eval': dataset_config.paths[dataset_config.train_eval_split]
          },
          feature_description=dataset_config.features),
      output_features=output_features,
      preprocessors=gen_task_preprocessors(training=True),
      postprocess_fn=partial(postprocessors.make_output_dict, audio_codec),
      metric_fns=metric_fns,
  )

  # Add eval tasks.
  for split in dataset_config.infer_eval_splits:
    eval_task_name = construct_task_name(
        task_prefix=task_prefix,
        dataset_name=dataset_config.name,
        audio_codec=audio_codec,
        vocab_config=vocab_config,
        note_representation_config=note_representation_config,
        task_suffix=split.suffix)

    if split.include_in_mixture:
      mixture_task_names.append(eval_task_name)

    seqio.TaskRegistry.add(
        eval_task_name,
        source=seqio.TFExampleDataSource(
            split_to_filepattern={'eval': dataset_config.paths[split.name]},
            feature_description=dataset_config.features),
        output_features=output_features,
        preprocessors=gen_task_preprocessors(training=False),
        postprocess_fn=partial(postprocessors.make_output_dict, audio_codec),
        metric_fns=metric_fns,
    )

    # Also construct a full song eval that will *not* be added to the mixture.
    full_eval_task_name = construct_task_name(
        task_prefix=task_prefix + '_full',
        dataset_name=dataset_config.name,
        audio_codec=audio_codec,
        vocab_config=vocab_config,
        note_representation_config=note_representation_config,
        task_suffix=split.suffix)
    seqio.TaskRegistry.add(
        full_eval_task_name,
        source=seqio.TFExampleDataSource(
            split_to_filepattern={'eval': dataset_config.paths[split.name]},
            feature_description=dataset_config.features),
        output_features=output_features,
        preprocessors=gen_task_preprocessors(
            training=False, full_song_eval=True),
        postprocess_fn=partial(postprocessors.make_output_dict, audio_codec),
        metric_fns=metric_fns,
    )

  # Construct mixture.
  seqio.MixtureRegistry.add(
      construct_task_name(
          task_prefix=task_prefix,
          dataset_name=dataset_config.name,
          audio_codec=audio_codec,
          vocab_config=vocab_config,
          note_representation_config=note_representation_config,
          task_suffix='eval'),
      mixture_task_names,
      default_rate=1)


##### Synthesis Tasks ########
def add_synthesis_tasks_to_registry(
    dataset_config: datasets.DatasetConfig,
    audio_codec: audio_codecs.AudioCodec,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn,  # TODO(iansimon): add type signature
    note_representation_config: NoteRepresentationConfig,
    skip_too_long: bool = False
) -> None:
  """Add both types of synthesis tasks to registry."""
  add_synthesis_task_to_registry(
      dataset_config=dataset_config,
      audio_codec=audio_codec,
      vocab_config=vocab_config,
      tokenize_fn=tokenize_fn,
      note_representation_config=note_representation_config,
      skip_too_long=skip_too_long)
  add_synthesis_with_context_task_to_registry(
      dataset_config=dataset_config,
      audio_codec=audio_codec,
      vocab_config=vocab_config,
      tokenize_fn=tokenize_fn,
      note_representation_config=note_representation_config,
      skip_too_long=skip_too_long)


def add_synthesis_task_to_registry(
    dataset_config: datasets.DatasetConfig,
    audio_codec: audio_codecs.AudioCodec,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn,  # TODO(iansimon): add type signature
    note_representation_config: NoteRepresentationConfig,
    skip_too_long: bool
) -> None:
  """Add note transcription task to seqio.TaskRegistry."""
  codec = vocabularies.build_codec(vocab_config)
  vocabulary = vocabularies.vocabulary_from_codec(codec)

  output_features = {
      'inputs': seqio.Feature(vocabulary=vocabulary),
      'targets': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
  }

  def gen_task_preprocessors(
      training: bool,
      full_song_eval: bool = False
  ) -> Sequence[Callable[..., tf.data.Dataset]]:
    assert not (training and full_song_eval)

    task_preprocessors = []
    task_preprocessors.extend(
        pre_cache_processor_chain(
            audio_codec=audio_codec,
            codec=codec,
            tokenize_fn=tokenize_fn,
            note_representation_config=note_representation_config,
            split_sequences=not full_song_eval))
    task_preprocessors.append(seqio.CacheDatasetPlaceholder())
    if full_song_eval:
      # Split entire sequence into segments.
      task_preprocessors.extend(
          split_full_song_processor_chain(audio_codec=audio_codec))
    else:
      # Select one random chunk.
      task_preprocessors.append(
          partial(
              t5.data.preprocessors.select_random_chunk,
              feature_key='targets',
              additional_feature_keys=[
                  'event_start_indices',
                  'event_end_indices',
                  'state_event_indices'
              ],
              passthrough_feature_keys=['inputs', 'state_events'],
              uniform_random_start=True))
    task_preprocessors.extend(
        note_representation_processor_chain(
            codec=codec,
            note_representation_config=note_representation_config))

    if training:
      allow_skip = skip_too_long
    else:
      allow_skip = False

    task_preprocessors.extend([
        partial(
            preprocessors.encode_audio,
            targets_keys=['targets'],
            keys_to_pad=['targets'] if training else None,
            audio_codec=audio_codec),
        partial(preprocessors.handle_too_long, skip=allow_skip),
        # Note: seqio.evalation requires copying pretokenized targets.
        partial(
            seqio.preprocessors.tokenize_and_append_eos,
            copy_pretokenized=True),
    ])
    return task_preprocessors  # pytype: disable=bad-return-type

  metric_fns = [
      partial(metrics.image_metric_fn, audio_codec=audio_codec),
      partial(metrics.audio_metric_fn, audio_codec=audio_codec),
      metrics.reconstruction_metric_fn,
      metrics.count_examples,
      metrics.model_timing,
      metrics.transcription_metric_fn
  ]

  construct_train_eval_tasks_and_mixture(
      task_prefix='synthesis',
      dataset_config=dataset_config,
      audio_codec=audio_codec,
      vocab_config=vocab_config,
      note_representation_config=note_representation_config,
      output_features=output_features,
      gen_task_preprocessors=gen_task_preprocessors,
      metric_fns=metric_fns)


def add_synthesis_with_context_task_to_registry(
    dataset_config: datasets.DatasetConfig,
    audio_codec: audio_codecs.AudioCodec,
    vocab_config: vocabularies.VocabularyConfig,
    tokenize_fn,  # TODO(iansimon): add type signature
    note_representation_config: NoteRepresentationConfig,
    skip_too_long: bool
) -> None:
  """Add note transcription task to seqio.TaskRegistry."""
  codec = vocabularies.build_codec(vocab_config)
  vocabulary = vocabularies.vocabulary_from_codec(codec)

  output_features = {
      'inputs': seqio.Feature(vocabulary=vocabulary),
      'targets': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
      'targets_context': seqio.ContinuousFeature(dtype=tf.float32, rank=2),
  }

  def gen_task_preprocessors(
      training: bool,
      full_song_eval: bool = False
  ) -> Sequence[Callable[..., tf.data.Dataset]]:
    assert not (training and full_song_eval)

    task_preprocessors = []
    task_preprocessors.extend(
        pre_cache_processor_chain(
            audio_codec=audio_codec,
            codec=codec,
            tokenize_fn=tokenize_fn,
            note_representation_config=note_representation_config,
            split_sequences=not full_song_eval))
    task_preprocessors.append(seqio.CacheDatasetPlaceholder())
    if full_song_eval:
      # Split entire sequence into segments and add empty context.
      # (Context will be filled in during inference.)
      task_preprocessors.extend(
          split_full_song_processor_chain(
              audio_codec=audio_codec,
              feature_context_key='targets_context'))
    else:
      # Select random chunk from sequence, with preceding context.
      task_preprocessors.append(
          partial(
              preprocessors.select_random_chunk_with_feature_context,
              audio_codec=audio_codec,
              feature_key='targets',
              feature_context_key='targets_context',
              additional_feature_keys=[
                  'event_start_indices',
                  'event_end_indices',
                  'state_event_indices'
              ],
              passthrough_feature_keys=['inputs', 'state_events']))
    task_preprocessors.extend(
        note_representation_processor_chain(
            codec=codec,
            note_representation_config=note_representation_config))

    if training:
      allow_skip = skip_too_long
    else:
      allow_skip = False

    task_preprocessors.extend([
        partial(
            preprocessors.encode_audio,
            targets_keys=['targets'],
            context_keys=['targets_context'],
            keys_to_pad=['targets'] if training else None,
            audio_codec=audio_codec),
        partial(preprocessors.handle_too_long, skip=allow_skip),
        # Note: seqio.evalation requires copying pretokenized targets.
        partial(
            seqio.preprocessors.tokenize_and_append_eos,
            copy_pretokenized=True),
    ])
    return task_preprocessors  # pytype: disable=bad-return-type

  metric_fns = [
      partial(metrics.image_metric_fn, audio_codec=audio_codec),
      partial(metrics.audio_metric_fn, audio_codec=audio_codec,
              context_keys=['targets_context']),
      metrics.reconstruction_metric_fn,
      metrics.count_examples,
      metrics.model_timing,
      metrics.transcription_metric_fn
  ]

  construct_train_eval_tasks_and_mixture(
      task_prefix='synthesis_with_context',
      dataset_config=dataset_config,
      audio_codec=audio_codec,
      vocab_config=vocab_config,
      note_representation_config=note_representation_config,
      output_features=output_features,
      gen_task_preprocessors=gen_task_preprocessors,
      metric_fns=metric_fns)


# ------------------------------------------------------------------------------
# --------------------- Actually Add Tasks to Registry -------------------------
# ------------------------------------------------------------------------------

# Create two vocabulary configs, one default and one with only on-off velocity.
VOCAB_CONFIG_FULL = vocabularies.VocabularyConfig()
VOCAB_CONFIG_NOVELOCITY = vocabularies.VocabularyConfig(num_velocity_bins=1)


SYNTH_MIXTURE_DATASET_CONFIGS = [
    datasets.MAESTROV3_CONFIG,
    datasets.GUITARSET_CONFIG,
    datasets.URMP_CONFIG,
    datasets.MUSICNET_CONFIG,
    datasets.CERBERUS4_CONFIG,
    datasets.SLAKH_CONFIG,
]

AUDIO_CODECS = [
    audio_codecs.MelGAN(),
    audio_codecs.SoundStream(),
    audio_codecs.MelWaveform(),
]

# -------------------------- Synthesis Task ------------------------------------
# Loop over audio codecs.

for ac in AUDIO_CODECS:
  # Synthesize MAESTRO v3.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.MAESTROV3_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_FULL,
      tokenize_fn=partial(
          preprocessors.tokenize_transcription_example,
          audio_is_samples=False,
          id_feature_key='id'),
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=False))

  # Synthesize MAESTRO v3 without velocities, with ties.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.MAESTROV3_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_NOVELOCITY,
      tokenize_fn=partial(
          preprocessors.tokenize_transcription_example,
          audio_is_samples=False,
          id_feature_key='id'),
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=True))

  # Synthesize GuitarSet, with ties.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.GUITARSET_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_NOVELOCITY,
      tokenize_fn=preprocessors.tokenize_guitarset_example,
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=True))

  # Synthesize URMP mixes, with ties.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.URMP_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_NOVELOCITY,
      tokenize_fn=partial(
          preprocessors.tokenize_example_with_program_lookup,
          inst_name_to_program_fn=preprocessors.urmp_instrument_to_program,
          id_feature_key='id'),
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=True))

  # Synthesize MusicNet, with ties.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.MUSICNET_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_NOVELOCITY,
      tokenize_fn=partial(
          preprocessors.tokenize_transcription_example,
          audio_is_samples=True,
          id_feature_key='id'),
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=True))

  # Synthesize Cerberus4 (piano-guitar-bass-drums quartets), with ties.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.CERBERUS4_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_NOVELOCITY,
      tokenize_fn=partial(
          preprocessors.tokenize_slakh_example,
          track_specs=datasets.CERBERUS4_CONFIG.track_specs,
          ignore_pitch_bends=True),
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=True))

  # Synthesize 10 random sub-mixes of each song from Slakh, with ties.
  add_synthesis_tasks_to_registry(
      dataset_config=datasets.SLAKH_CONFIG,
      audio_codec=ac,
      vocab_config=VOCAB_CONFIG_NOVELOCITY,
      tokenize_fn=partial(
          preprocessors.tokenize_slakh_example,
          track_specs=None,
          ignore_pitch_bends=True),
      note_representation_config=NoteRepresentationConfig(
          onsets_only=False, include_ties=True))

  def add_synthesis_mixture(task_prefix,
                            mixing_temperature=10/3,
                            audio_codec=ac):
    """Add synthesis mixture task for the given task prefix."""
    synth_mixture_train_task_names = []
    synth_mixture_eval_task_names = []
    for dataset_cfg in SYNTH_MIXTURE_DATASET_CONFIGS:
      synth_mixture_train_task_names.append(
          construct_task_name(
              task_prefix=task_prefix,
              dataset_name=dataset_cfg.name,
              audio_codec=audio_codec,
              vocab_config=VOCAB_CONFIG_NOVELOCITY,
              note_representation_config=NoteRepresentationConfig(
                  onsets_only=False, include_ties=True),
              task_suffix='train'))
      synth_mixture_eval_task_names.append(
          construct_task_name(
              task_prefix=task_prefix,
              dataset_name=dataset_cfg.name,
              audio_codec=audio_codec,
              vocab_config=VOCAB_CONFIG_NOVELOCITY,
              note_representation_config=NoteRepresentationConfig(
                  onsets_only=False, include_ties=True),
              task_suffix='validation'))

    logging.info('Adding mixture with task names %s',
                 synth_mixture_train_task_names)

    # Add the mixture of all synthesis tasks, with ties.
    seqio.MixtureRegistry.add(
        construct_task_name(
            task_prefix=task_prefix,
            dataset_name='mega',
            audio_codec=audio_codec,
            vocab_config=VOCAB_CONFIG_NOVELOCITY,
            note_representation_config=NoteRepresentationConfig(
                onsets_only=False, include_ties=True),
            task_suffix='train'),
        synth_mixture_train_task_names,
        default_rate=partial(
            seqio.mixing_rate_num_examples,
            temperature=mixing_temperature))
    seqio.MixtureRegistry.add(
        construct_task_name(
            task_prefix=task_prefix,
            dataset_name='mega',
            audio_codec=audio_codec,
            vocab_config=VOCAB_CONFIG_NOVELOCITY,
            note_representation_config=NoteRepresentationConfig(
                onsets_only=False, include_ties=True),
            task_suffix='eval'),
        synth_mixture_eval_task_names,
        default_rate=partial(
            seqio.mixing_rate_num_examples,
            temperature=mixing_temperature))

  add_synthesis_mixture('synthesis')
  add_synthesis_mixture('synthesis_with_context')
