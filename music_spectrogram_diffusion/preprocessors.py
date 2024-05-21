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

"""Transcription preprocessors."""

import itertools
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
import gin
from immutabledict import immutabledict
import librosa
from music_spectrogram_diffusion import audio_codecs
from music_spectrogram_diffusion import event_codec
from music_spectrogram_diffusion import note_sequences
from music_spectrogram_diffusion import run_length_encoding
from music_spectrogram_diffusion import vocabularies
import note_seq
import numpy as np
import seqio
import t5
import tensorflow as tf


def add_unique_id(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Add unique integer ID to each example in a dataset."""
  def add_id_field(i, ex):
    ex['unique_id'] = [i]
    return ex
  return ds.enumerate().map(
      add_id_field, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@seqio.map_over_dataset
def pad_notesequence_array(ex, feature_key='input_times'):
  """Pad the NoteSequence array so that it can later be "split"."""
  ex['sequence'] = tf.pad(tf.expand_dims(ex['sequence'], 0),
                          [[0, len(ex[feature_key]) - 1]])
  return ex


@seqio.map_over_dataset
def add_dummy_targets(ex):
  """Add dummy targets; used in eval when targets are not actually used."""
  ex['targets'] = np.array([], dtype=np.int32)
  return ex


def _audio_to_frames(
    samples: Sequence[float],
    hop_size: int,
    frame_rate: int,
) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
  """Convert audio samples to non-overlapping frames and frame times."""
  frame_size = hop_size
  logging.info('Padding %d samples to multiple of %d', len(samples), frame_size)
  samples = np.pad(samples,
                   [0, frame_size - len(samples) % frame_size],
                   mode='constant')

  # Split audio into frames.
  frames = tf.signal.frame(
      samples, frame_length=frame_size, frame_step=frame_size, pad_end=True)

  num_frames = len(samples) // frame_size
  logging.info('Encoded %d samples to %d frames (%d samples each)',
               len(samples), num_frames, frame_size)

  times = np.arange(num_frames) / frame_rate
  return frames, times


def _flatten_frames(frames):
  """Convert frames back into a flat array of samples."""
  return tf.reshape(frames, [-1])


def _include_inputs(ds, input_record, fields_to_omit=('audio',)):
  """Include fields from input record (other than audio) in dataset records."""
  def include_inputs_fn(output_record):
    for key in set(input_record.keys()) - set(output_record.keys()):
      output_record[key] = input_record[key]
    for key in fields_to_omit:
      del output_record[key]
    return output_record
  return ds.map(include_inputs_fn,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


def tokenize_transcription_example(
    ds: tf.data.Dataset,
    audio_codec: audio_codecs.AudioCodec,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    audio_is_samples: bool,
    id_feature_key: Optional[str] = None) -> tf.data.Dataset:
  """Tokenize a note transcription example for run-length encoding.

  Outputs include:
    inputs: audio sample frames, num_frames-by-frame_size
    input_time: timestamp for each frame
    targets: symbolic sequence of note-related events
    event_start_indices: start target index for every input index
    event_end_indices: end target index for every input index

  Args:
    ds: Input dataset.
    audio_codec: Spectrogram configuration.
    codec: Event vocabulary codec.
    is_training_data: Unused.
    onsets_only: If True, include only onset events (not offset, velocity, or
        program).
    include_ties: If True, also write state events containing active notes to
        support a "tie" section after run-length encoding.
    audio_is_samples: If True, audio is floating-point samples instead of
        serialized WAV.
    id_feature_key: If not None, replace sequence ID with specified key field
        from the dataset.

  Returns:
    Dataset with the outputs described above.
  """
  del is_training_data

  if onsets_only and include_ties:
    raise ValueError('Ties not supported when only modeling onsets.')

  def tokenize(sequence, audio, sample_rate, example_id=None):
    ns = note_seq.NoteSequence.FromString(sequence)
    note_sequences.validate_note_sequence(ns)

    if example_id is not None:
      ns.id = example_id

    if audio_is_samples:
      samples = audio
      if sample_rate != audio_codec.sample_rate:
        samples = librosa.resample(
            samples, sample_rate, audio_codec.sample_rate)
    else:
      samples = note_seq.audio_io.wav_data_to_samples_librosa(
          audio, sample_rate=audio_codec.sample_rate)

    logging.info('Got samples for %s::%s with length %d',
                 ns.id, ns.filename, len(samples))

    frames, frame_times = _audio_to_frames(samples,
                                           audio_codec.hop_size,
                                           audio_codec.frame_rate)

    if onsets_only:
      times, values = note_sequences.note_sequence_to_onsets(ns)
    else:
      ns = note_seq.apply_sustain_control_changes(ns)
      times, values = (
          note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

    # The original NoteSequence can have a lot of control changes we don't need;
    # delete them.
    del ns.control_changes[:]

    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = (
         run_length_encoding.encode_and_index_events(
             state=note_sequences.NoteEncodingState() if include_ties else None,
             event_times=times,
             event_values=values,
             encode_event_fn=note_sequences.note_event_data_to_events,
             codec=codec,
             frame_times=frame_times,
             encoding_state_to_events_fn=(
                 note_sequences.note_encoding_state_to_events
                 if include_ties else None)))

    yield {
        'inputs': frames,
        'input_times': frame_times,
        'targets': events,
        'event_start_indices': event_start_indices,
        'event_end_indices': event_end_indices,
        'state_events': state_events,
        'state_event_indices': state_event_indices,
        'sequence': ns.SerializeToString()
    }

  def process_record(input_record):
    if audio_is_samples and 'sample_rate' not in input_record:
      raise ValueError('Must provide sample rate when audio is samples.')

    args = [
        input_record['sequence'],
        input_record['audio'],
        input_record['sample_rate'] if 'sample_rate' in input_record else 0
    ]
    if id_feature_key is not None:
      args.append(input_record[id_feature_key])

    ds = tf.data.Dataset.from_generator(
        tokenize,
        output_signature={
            'inputs':
                tf.TensorSpec(
                    shape=(None, audio_codec.hop_size),
                    dtype=tf.float32),
            'input_times':
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'event_start_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'event_end_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_events':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_event_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'sequence':
                tf.TensorSpec(shape=(), dtype=tf.string)
        },
        args=args)

    ds = _include_inputs(ds, input_record)
    return ds

  tokenized_records = ds.flat_map(process_record)
  return tokenized_records


def tokenize_guitarset_example(
    ds: tf.data.Dataset,
    audio_codec: audio_codecs.AudioCodec,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool
) -> tf.data.Dataset:
  """Tokenize a GuitarSet transcription example."""
  def _preprocess_example(ex, name):
    assert 'inst_names' not in ex, 'Key `inst_names` is already populated.'
    ex['inst_names'] = [name]
    ex['instrument_sequences'] = [ex.pop('sequence')]
    return ex

  ds = ds.map(
      lambda x: _preprocess_example(x, 'Clean Guitar'),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = tokenize_example_with_program_lookup(
      ds,
      audio_codec=audio_codec,
      codec=codec,
      is_training_data=is_training_data,
      inst_name_to_program_fn=guitarset_instrument_to_program,
      onsets_only=onsets_only,
      include_ties=include_ties,
      id_feature_key='id')
  return ds


def guitarset_instrument_to_program(instrument: str) -> int:
  """GuitarSet is all guitar, return the first MIDI guitar program."""
  if instrument == 'Clean Guitar':
    return 24
  else:
    raise ValueError('Unknown GuitarSet instrument: %s' % instrument)


def tokenize_example_with_program_lookup(
    ds: tf.data.Dataset,
    audio_codec: audio_codecs.AudioCodec,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    inst_name_to_program_fn: Callable[[str], int],
    id_feature_key: Optional[str] = None
) -> tf.data.Dataset:
  """Tokenize an example, optionally looking up and assigning program numbers.

  This can be used by any dataset where a mapping function can be used to
  map from the inst_names feature to a set of program numbers.

  Args:
    ds: Input dataset.
    audio_codec: Spectrogram configuration.
    codec: Event vocabulary codec.
    is_training_data: Unused.
    onsets_only: If True, include only onset events (not offset & velocity).
    include_ties: If True, include tie events.
    inst_name_to_program_fn: A function used to map the instrument names
      in the `inst_names` feature of each example to a MIDI program number.
    id_feature_key: If not None, replace sequence ID with specified key field
        from the dataset.

  Returns:
    Dataset with the outputs described above.
  """
  del is_training_data

  def tokenize(sequences, inst_names, audio, example_id=None):
    # Add all the notes from the tracks to a single NoteSequence.
    ns = note_seq.NoteSequence(ticks_per_quarter=220)
    tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
    assert len(tracks) == len(inst_names)
    for track, inst_name in zip(tracks, inst_names):
      program = inst_name_to_program_fn(
          inst_name.decode())

      # Note that there are no pitch bends in URMP data; the below block will
      # raise PitchBendError if one is encountered.
      add_track_to_notesequence(ns, track, program=program, is_drum=False,
                                ignore_pitch_bends=False)

    note_sequences.assign_instruments(ns)
    note_sequences.validate_note_sequence(ns)

    if example_id is not None:
      ns.id = example_id

    samples = note_seq.audio_io.wav_data_to_samples_librosa(
        audio, sample_rate=audio_codec.sample_rate)

    logging.info('Got samples for %s::%s with length %d',
                 ns.id, ns.filename, len(samples))

    frames, frame_times = _audio_to_frames(samples,
                                           audio_codec.hop_size,
                                           audio_codec.frame_rate)

    if onsets_only:
      times, values = note_sequences.note_sequence_to_onsets(ns)
    else:
      times, values = (
          note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

    # The original NoteSequence can have a lot of control changes we don't need;
    # delete them.
    del ns.control_changes[:]

    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = (
         run_length_encoding.encode_and_index_events(
             state=note_sequences.NoteEncodingState() if include_ties else None,
             event_times=times,
             event_values=values,
             encode_event_fn=note_sequences.note_event_data_to_events,
             codec=codec,
             frame_times=frame_times,
             encoding_state_to_events_fn=(
                 note_sequences.note_encoding_state_to_events
                 if include_ties else None)))

    yield {
        'inputs': frames,
        'input_times': frame_times,
        'targets': events,
        'event_start_indices': event_start_indices,
        'event_end_indices': event_end_indices,
        'state_events': state_events,
        'state_event_indices': state_event_indices,
        'sequence': ns.SerializeToString()
    }

  def process_record(input_record):
    args = [
        input_record['instrument_sequences'],
        input_record['inst_names'],
        input_record['audio'],
    ]
    if id_feature_key is not None:
      args.append(input_record[id_feature_key])

    ds = tf.data.Dataset.from_generator(
        tokenize,
        output_signature={
            'inputs':
                tf.TensorSpec(
                    shape=(None, audio_codec.hop_size),
                    dtype=tf.float32),
            'input_times':
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'event_start_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'event_end_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_events':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_event_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'sequence':
                tf.TensorSpec(shape=(), dtype=tf.string)
        },
        args=args)

    ds = _include_inputs(ds, input_record)
    return ds

  tokenized_records = ds.flat_map(process_record)
  return tokenized_records


_URMP_INSTRUMENT_PROGRAMS = immutabledict({
    'vn': 40,   # violin
    'va': 41,   # viola
    'vc': 42,   # cello
    'db': 43,   # double bass
    'tpt': 56,  # trumpet
    'tbn': 57,  # trombone
    'tba': 58,  # tuba
    'hn': 60,   # French horn
    'sax': 64,  # saxophone
    'ob': 68,   # oboe
    'bn': 70,   # bassoon
    'cl': 71,   # clarinet
    'fl': 73    # flute
})


def urmp_instrument_to_program(urmp_instrument: str) -> int:
  """Fetch the program number associated with a given URMP instrument code."""
  if urmp_instrument not in _URMP_INSTRUMENT_PROGRAMS:
    raise ValueError('unknown URMP instrument: %s' % urmp_instrument)
  return _URMP_INSTRUMENT_PROGRAMS[urmp_instrument]


_SLAKH_CLASS_PROGRAMS = immutabledict({
    'Acoustic Piano': 0,
    'Electric Piano': 4,
    'Chromatic Percussion': 8,
    'Organ': 16,
    'Acoustic Guitar': 24,
    'Clean Electric Guitar': 26,
    'Distorted Electric Guitar': 29,
    'Acoustic Bass': 32,
    'Electric Bass': 33,
    'Violin': 40,
    'Viola': 41,
    'Cello': 42,
    'Contrabass': 43,
    'Orchestral Harp': 46,
    'Timpani': 47,
    'String Ensemble': 48,
    'Synth Strings': 50,
    'Choir and Voice': 52,
    'Orchestral Hit': 55,
    'Trumpet': 56,
    'Trombone': 57,
    'Tuba': 58,
    'French Horn': 60,
    'Brass Section': 61,
    'Soprano/Alto Sax': 64,
    'Tenor Sax': 66,
    'Baritone Sax': 67,
    'Oboe': 68,
    'English Horn': 69,
    'Bassoon': 70,
    'Clarinet': 71,
    'Pipe': 73,
    'Synth Lead': 80,
    'Synth Pad': 88
})


def slakh_class_to_program_and_is_drum(slakh_class: str) -> Tuple[int, bool]:
  """Map Slakh class string to program number and boolean indicating drums."""
  if slakh_class == 'Drums':
    return 0, True
  elif slakh_class not in _SLAKH_CLASS_PROGRAMS:
    raise ValueError('unknown Slakh class: %s' % slakh_class)
  else:
    return _SLAKH_CLASS_PROGRAMS[slakh_class], False


class PitchBendError(Exception):
  pass


def add_track_to_notesequence(ns: note_seq.NoteSequence,
                              track: note_seq.NoteSequence,
                              program: int, is_drum: bool,
                              ignore_pitch_bends: bool):
  """Add a track to a NoteSequence."""
  if track.pitch_bends and not ignore_pitch_bends:
    raise PitchBendError
  track_sus = note_seq.apply_sustain_control_changes(track)
  for note in track_sus.notes:
    note.program = program
    note.is_drum = is_drum
    ns.notes.extend([note])
    ns.total_time = max(ns.total_time, note.end_time)


def tokenize_slakh_example(
    ds: tf.data.Dataset,
    audio_codec: audio_codecs.AudioCodec,
    codec: event_codec.Codec,
    is_training_data: bool,
    onsets_only: bool,
    include_ties: bool,
    track_specs: Optional[Sequence[note_sequences.TrackSpec]],
    ignore_pitch_bends: bool
) -> tf.data.Dataset:
  """Tokenize a Slakh multitrack note transcription example."""
  def tokenize(sequences, samples, sample_rate, inst_names, example_id):
    if sample_rate != audio_codec.sample_rate:
      samples = librosa.resample(
          samples, sample_rate, audio_codec.sample_rate)

    frames, frame_times = _audio_to_frames(samples,
                                           audio_codec.hop_size,
                                           audio_codec.frame_rate)

    # Add all the notes from the tracks to a single NoteSequence.
    ns = note_seq.NoteSequence(ticks_per_quarter=220)
    tracks = [note_seq.NoteSequence.FromString(seq) for seq in sequences]
    assert len(tracks) == len(inst_names)
    if track_specs:
      # Specific tracks expected.
      assert len(tracks) == len(track_specs)
      for track, spec, inst_name in zip(tracks, track_specs, inst_names):
        # Make sure the instrument name matches what we expect.
        assert inst_name.decode() == spec.name
        try:
          add_track_to_notesequence(ns, track,
                                    program=spec.program, is_drum=spec.is_drum,
                                    ignore_pitch_bends=ignore_pitch_bends)
        except PitchBendError:
          # TODO(iansimon): is there a way to count these?
          return
    else:
      for track, inst_name in zip(tracks, inst_names):
        # Instrument name should be Slakh class.
        program, is_drum = slakh_class_to_program_and_is_drum(
            inst_name.decode())
        try:
          add_track_to_notesequence(ns, track, program=program, is_drum=is_drum,
                                    ignore_pitch_bends=ignore_pitch_bends)
        except PitchBendError:
          # TODO(iansimon): is there a way to count these?
          return

    note_sequences.assign_instruments(ns)
    note_sequences.validate_note_sequence(ns)
    if is_training_data:
      # Trim overlapping notes in training (as our event vocabulary cannot
      # represent them), but preserve original NoteSequence for eval.
      ns = note_sequences.trim_overlapping_notes(ns)

    ns.id = example_id

    if onsets_only:
      times, values = note_sequences.note_sequence_to_onsets(ns)
    else:
      times, values = (
          note_sequences.note_sequence_to_onsets_and_offsets_and_programs(ns))

    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = (
         run_length_encoding.encode_and_index_events(
             state=note_sequences.NoteEncodingState() if include_ties else None,
             event_times=times,
             event_values=values,
             encode_event_fn=note_sequences.note_event_data_to_events,
             codec=codec,
             frame_times=frame_times,
             encoding_state_to_events_fn=(
                 note_sequences.note_encoding_state_to_events
                 if include_ties else None)))

    yield {
        'inputs': frames,
        'input_times': frame_times,
        'targets': events,
        'event_start_indices': event_start_indices,
        'event_end_indices': event_end_indices,
        'state_events': state_events,
        'state_event_indices': state_event_indices,
        'sequence': ns.SerializeToString()
    }

  def process_record(input_record):
    ds = tf.data.Dataset.from_generator(
        tokenize,
        output_signature={
            'inputs':
                tf.TensorSpec(
                    shape=(None, audio_codec.hop_size),
                    dtype=tf.float32),
            'input_times':
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            'targets':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'event_start_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'event_end_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_events':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'state_event_indices':
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'sequence':
                tf.TensorSpec(shape=(), dtype=tf.string)
        },
        args=[
            input_record['note_sequences'], input_record['mix'],
            input_record['audio_sample_rate'], input_record['inst_names'],
            input_record['track_id']
        ])

    ds = _include_inputs(ds, input_record, fields_to_omit=['mix', 'stems'])
    return ds

  tokenized_records = ds.flat_map(process_record)
  return tokenized_records


def encode_audio(
    dataset: tf.data.Dataset,
    output_features: seqio.preprocessors.OutputFeaturesType,
    sequence_length: seqio.preprocessors.SequenceLengthType,
    audio_codec: audio_codecs.AudioCodec,
    targets_keys: Optional[Sequence[str]] = (),
    context_keys: Optional[Sequence[str]] = (),
    keys_to_pad: Optional[Sequence[str]] = None,
) -> tf.data.Dataset:
  """Encodes audio from frames of samples using an audio codec.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key feature_key.
    output_features: Mapping of keys to features.
    sequence_length: Used to determine the length for feature_key and
      feature_context_key.
    audio_codec: Codec to use for encoding.
    targets_keys: Which features to encode as targets.
    context_keys: Which features to encode as context.
    keys_to_pad: If specified, these keys will be padded with silence to the
      full length of the feature.

  Returns:
    a dataset
  """

  @seqio.map_over_dataset
  def encode(ex):
    """Encode audio."""
    for k in list(targets_keys) + list(context_keys):
      # Use separate codec for audio context.
      ac = audio_codec.context_codec if k in context_keys else audio_codec

      frames = ex[k]

      max_feature_length = sequence_length[k]

      if output_features[k].add_eos:
        # Leave room to insert an EOS token.
        max_feature_length -= 1

      # Additional frames may be added for STFT purposes.
      tf.debugging.assert_less_equal(
          tf.shape(frames)[0],
          max_feature_length + ac.additional_frames_for_encoding)

      if keys_to_pad and k in keys_to_pad:
        padding = tf.maximum(0, max_feature_length - tf.shape(frames)[0])
        frames = tf.pad(frames, [[0, padding], [0, 0]])

      # Slice off any additional STFT frames.
      samples = _flatten_frames(frames[:max_feature_length])
      ex[f'raw_{k}'] = samples

      # Do encoding on the full set of frames.
      encoded = ac.encode(_flatten_frames(frames))
      tf.assert_equal(tf.shape(frames)[0], tf.shape(encoded)[0],
                      f'Length of {k} was not the same before and after '
                      'encoding.')
      # Now slice off any extra.
      encoded = encoded[:max_feature_length]
      ex[k] = encoded
    return ex

  dataset = encode(dataset)  # pylint: disable=no-value-for-parameter
  return dataset


def handle_too_long(dataset: tf.data.Dataset,
                    output_features: seqio.preprocessors.OutputFeaturesType,
                    sequence_length: seqio.preprocessors.SequenceLengthType,
                    skip: bool = False) -> tf.data.Dataset:
  """Handle sequences that are too long, by either failing or skipping them."""
  def max_length_for_key(key):
    max_length = sequence_length[key]
    if output_features[key].add_eos:
      max_length -= 1
    return max_length

  if skip:
    # Drop examples where one of the features is longer than its maximum
    # sequence length.
    def is_not_too_long(ex):
      return not tf.reduce_any(
          [k in output_features and len(v) > max_length_for_key(k)
           for k, v in ex.items()])
    dataset = dataset.filter(is_not_too_long)

  def assert_not_too_long(key: str, value: tf.Tensor) -> tf.Tensor:
    if key in output_features:
      max_length = max_length_for_key(key)
      tf.debugging.assert_less_equal(
          tf.shape(value)[0], max_length,
          f'Value for "{key}" field exceeds maximum length')
    return value

  # Assert that no examples have features longer than their maximum sequence
  # length.
  return dataset.map(
      lambda ex: {k: assert_not_too_long(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable
def map_midi_programs(
    ds: tf.data.Dataset,
    codec: event_codec.Codec,
    granularity_type: str = 'full',
    feature_key: str = 'targets'
) -> Mapping[str, Any]:
  """Apply MIDI program map to token sequences."""
  granularity = vocabularies.PROGRAM_GRANULARITIES[granularity_type]
  def _map_program_tokens(ex):
    ex[feature_key] = granularity.tokens_map_fn(ex[feature_key], codec)
    return ex

  return ds.map(
      _map_program_tokens, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def select_random_chunk_with_feature_context(
    dataset: tf.data.Dataset,
    output_features: seqio.preprocessors.OutputFeaturesType,
    sequence_length: seqio.preprocessors.SequenceLengthType,
    feature_key: str,
    feature_context_key: str,
    audio_codec: audio_codecs.AudioCodec,
    additional_feature_keys: Optional[Sequence[str]] = None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
    minimum_target_frames: int = 1) -> tf.data.Dataset:
  """Select a random chunk of feature_key with context in feature_context_key.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key feature_key.
    output_features: Mapping of keys to features.
    sequence_length: Used to determine the length for feature_key and
      feature_context_key.
    feature_key: Which feature to use from the dataset.
    feature_context_key: Where to store the context.
    audio_codec: AudioCodec configuration.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length. Note that these keys will not
      have the 'context' portion saved.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    minimum_target_frames: Minimum number of frames to include in the target.

  Returns:
    a dataset
  """
  assert minimum_target_frames >= 1
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'chunk keys {overlap_keys} also included in passthrough keys')

  max_feature_length = sequence_length[feature_key]
  max_context_length = sequence_length[feature_context_key]

  if output_features[feature_key].add_eos:
    # Leave room to insert an EOS token.
    max_feature_length -= 1

  if output_features[feature_context_key].add_eos:
    # Leave room to insert an EOS token.
    max_context_length -= 1

  @seqio.map_over_dataset(num_seeds=1)
  def select_chunk(x, seed):
    """Select a random chunk of tokens."""
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    tf.debugging.assert_greater_equal(n_tokens, minimum_target_frames)
    start = tf.random.stateless_uniform(
        [],
        # minimum is no context, but still fill the regular buffer.
        # Example: length = 10, feature_length = 2, context_length = 3
        # start = -3
        # context_start = max(0, -3) = 0
        # context_end = -3 + 3 = 0
        # feature_start = 0
        # feature_end = min(0 + 2, 10) = 2
        minval=-max_context_length,
        # maximum is full context plus at least minimum_target_frames.
        # Example: length = 10, feature_length = 2, context_length = 3,
        #   minimum_target_frames = 1
        # start = 10 - 3 - 1 = 6
        # context_start = max(0, 6) = 6
        # context_end = 6 + 3 = 9
        # feature_start = 9
        # feature_end = min(9 + 2, 10) = 10
        #
        # Example: length = 1, minimum_target_frames = 1
        # start = 1 - 3 - 1 = -3
        # context_start = max(0, -3) = 0
        # context_end = -3 + 3 = 0
        # feature_start = 0
        # feature_end = min(0 + 2, 1) = 1
        maxval=n_tokens - max_context_length - minimum_target_frames,
        dtype=tf.int32,
        seed=seed)
    context_start = tf.maximum(0, start)
    context_end = start + max_context_length

    feature_start = context_end
    feature_end = tf.minimum(feature_start + max_feature_length, n_tokens)
    chunk = {
        feature_context_key:
            tokens[context_start:context_end +
                   audio_codec.context_codec.additional_frames_for_encoding],
        feature_key:
            tokens[feature_start:feature_end +
                   audio_codec.additional_frames_for_encoding],
    }
    if additional_feature_keys is not None:
      for k in additional_feature_keys:
        tf.debugging.assert_equal(
            tf.shape(tokens)[0], tf.shape(x[k])[0], message=(
                f'Additional feature {k} is not the same size as '
                f'{feature_key} along axis 0 in select_random_chunk().'))
        chunk[k] = x[k][feature_start:feature_end]
    if passthrough_feature_keys is not None:
      for k in passthrough_feature_keys:
        chunk[k] = x[k]
    return chunk
  dataset = select_chunk(dataset)  # pylint: disable=no-value-for-parameter

  return dataset


def split_full_song(
    dataset: tf.data.Dataset,
    output_features: seqio.preprocessors.OutputFeaturesType,
    sequence_length: seqio.preprocessors.SequenceLengthType,
    feature_key: str,
    audio_codec: audio_codecs.AudioCodec,
    additional_feature_keys: Optional[Sequence[str]] = None,
    passthrough_feature_keys: Optional[Sequence[str]] = None
) -> tf.data.Dataset:
  """Select a random chunk of feature_key with context in feature_context_key.

  Args:
    dataset: A tf.data.Dataset with dictionaries containing the key feature_key.
    output_features: Mapping of keys to features.
    sequence_length: Used to determine the length for feature_key and
      feature_context_key.
    feature_key: Which feature to use from the dataset for splitting.
    audio_codec: AudioCodec configuration.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length. Note that these keys will not
      have the 'context' portion saved.
    passthrough_feature_keys: Additional keys to pass through unchanged.

  Returns:
    a dataset
  """
  orig_feature_key = feature_key + '_orig'

  def feature_to_idxs(x):
    assert orig_feature_key not in x
    x[orig_feature_key] = x[feature_key]
    x[feature_key] = tf.range(tf.shape(x[feature_key])[0])
    return x
  dataset = dataset.map(feature_to_idxs, num_parallel_calls=tf.data.AUTOTUNE)

  max_tokens = sequence_length[feature_key]
  if output_features[feature_key].add_eos:
    # Leave room to insert an EOS token.
    max_tokens -= 1

  dataset = t5.data.preprocessors.split_tokens(
      dataset,
      max_tokens_per_segment=max_tokens,
      feature_key=feature_key,
      additional_feature_keys=additional_feature_keys,
      passthrough_feature_keys=list(itertools.chain(
          passthrough_feature_keys, [orig_feature_key])))

  def slice_feature_with_additional_frames(x):
    start = x[feature_key][0]
    end = x[feature_key][-1]
    end += audio_codec.additional_frames_for_encoding
    x[feature_key] = x[orig_feature_key][start:end]
    del x[orig_feature_key]
    return x
  dataset = dataset.map(slice_feature_with_additional_frames,
                        num_parallel_calls=tf.data.AUTOTUNE)
  return dataset
