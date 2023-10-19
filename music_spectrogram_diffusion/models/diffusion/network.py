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

"""T5.1.1 Transformer model."""

from typing import Any, Sequence

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp

from music_spectrogram_diffusion import layers
from music_spectrogram_diffusion.models.diffusion import diffusion_utils


def get_sequence_length(sequence: jnp.ndarray) -> jnp.ndarray:
  """Return the length of non-zero entries in the sequence."""
  # Return the first index where a 0 occurs.
  length = jnp.argmax(sequence == 0)

  # If argmax returns 0, that means that either
  # 1) No 0s were found, and the sequence length is the full length of the array
  # 2) There's padding immediately at the beginning, indicating that the array
  #    is all padding and the sequence length is 0.
  length = jnp.where(jnp.logical_and(length == 0, sequence[0] != 0),
                     sequence.shape[0], length)
  return length


def make_sequence_terminal_relative(pos_seq: jnp.ndarray,
                                    seq_len: jnp.ndarray) -> jnp.ndarray:
  """Convert positions or position encodings to terminal-relative coords."""
  # If a sequence has a max length of 5 and a sequence length of 2:
  # From: [0, 1, 2, 3, 4]
  # To:   [3, 4, 0, 1, 2]
  # Ensures that the final position is constant (in this case, always 4)
  # For relative positions, we can just resume the position counter at 6
  # for the next sequence and everything will line up.
  return jnp.roll(pos_seq, seq_len, axis=0)


@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  max_decoder_noise_time: float = 2e4
  decoder_cross_attend_style: str = 'sum_cross_attends'
  position_encoding: str = 'fixed'
  context_positions: str = 'regular'


def position_encoding_layer(config: T5Config, max_length: int):
  """Build position encoding layer based on the current config."""
  if config.position_encoding == 'fixed':
    return layers.Embed(
        num_embeddings=max_length,
        features=config.emb_dim,
        dtype=config.dtype,
        embedding_init=layers.sinusoidal(),
        fixed=True)
  elif config.position_encoding == 'fixed_permuted_offset':
    return layers.Embed(
        num_embeddings=max_length,
        features=config.emb_dim,
        dtype=config.dtype,
        embedding_init=layers.sinusoidal(
            permute_bands=True, random_phase_offsets=True),
        fixed=True)
  elif config.position_encoding == 'learnable_permuted_offset':
    return layers.Embed(
        num_embeddings=max_length,
        features=config.emb_dim,
        dtype=config.dtype,
        embedding_init=layers.sinusoidal(
            permute_bands=True, random_phase_offsets=True),
        fixed=False)
  elif config.position_encoding == 'random':
    return layers.Embed(
        num_embeddings=max_length,
        features=config.emb_dim,
        dtype=config.dtype)
  else:
    raise ValueError(f'Unknown position_encoding: {config.position_encoding}')


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: T5Config

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               encoder_inputs_mask: jnp.ndarray,
               deterministic: bool) -> jnp.ndarray:
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3

    # Make padding attention mask.
    encoder_mask = layers.make_attention_mask(
        encoder_inputs_mask, encoder_inputs_mask, dtype=cfg.dtype)

    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_attention_layer_norm')(
            inputs)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='attention')(
            x, x, encoder_mask, deterministic=deterministic)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    return y


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: T5Config

  @nn.compact
  def __call__(self,
               inputs,
               encodings_and_masks,
               conditioning_emb,
               deterministic=False):
    cfg = self.config

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(dtype=cfg.dtype,
                         name='pre_self_attention_layer_norm')(inputs)

    if conditioning_emb is not None:
      x = layers.FiLMLayer()(x, conditioning_emb)

    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='self_attention')(
            x,
            x,
            deterministic=deterministic)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # Encoder-Decoder blocks.
    y = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(x)

    if cfg.decoder_cross_attend_style == 'sum_cross_attends':
      ys = []
      for encoded, encoder_decoder_mask in encodings_and_masks:
        y_n = layers.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            head_dim=cfg.head_dim,
            dropout_rate=cfg.dropout_rate)(
                y,
                encoded,
                encoder_decoder_mask,
                deterministic=deterministic)
        y_n = layers.zero_activations_if_masked(y_n, encoder_decoder_mask)
        y_n = nn.Dropout(
            rate=cfg.dropout_rate, broadcast_dims=(-2,))(
                y_n, deterministic=deterministic)
        ys.append(y_n)
      y = sum(ys) + x
    elif cfg.decoder_cross_attend_style == 'concat_encodings':
      encoded = jnp.concatenate([x[0] for x in encodings_and_masks], axis=1)
      encoder_decoder_mask = jnp.concatenate(
          [x[1] for x in encodings_and_masks], axis=-1)

      y = layers.MultiHeadDotProductAttention(
          num_heads=cfg.num_heads,
          dtype=cfg.dtype,
          head_dim=cfg.head_dim,
          dropout_rate=cfg.dropout_rate)(
              y,
              encoded,
              encoder_decoder_mask,
              deterministic=deterministic)
      y = layers.zero_activations_if_masked(y, encoder_decoder_mask)
      y = nn.Dropout(
          rate=cfg.dropout_rate, broadcast_dims=(-2,))(
              y, deterministic=deterministic)
      y = y + x
    else:
      raise ValueError(f'Unknown decoder_cross_attend_style: '
                       f'{cfg.decoder_cross_attend_style}')

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)

    if conditioning_emb is not None:
      z = layers.FiLMLayer()(z, conditioning_emb)

    z = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y

    return z


class TokenEncoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config

  @nn.compact
  def __call__(self,
               encoder_input_tokens,
               encoder_inputs_mask,
               deterministic):
    cfg = self.config

    assert encoder_input_tokens.ndim == 2  # [batch, length]

    seq_length = encoder_input_tokens.shape[1]
    inputs_positions = jnp.arange(seq_length)[None, :]

    # [batch, length] -> [batch, length, emb_dim]
    x = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')(encoder_input_tokens.astype('int32'))

    x += position_encoding_layer(config=cfg, max_length=seq_length)(
        inputs_positions)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
          config=cfg,
          name=f'layers_{lyr}')(
              inputs=x,
              encoder_inputs_mask=encoder_inputs_mask,
              deterministic=deterministic)
    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
    return x, encoder_inputs_mask


class ContinuousEncoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config

  @nn.compact
  def __call__(self,
               encoder_inputs: jnp.ndarray,
               encoder_inputs_mask: jnp.ndarray,
               deterministic: bool):
    cfg = self.config

    assert encoder_inputs.ndim == 3  # [batch, length, input_dims]
    max_positions = encoder_inputs.shape[1]

    # [batch, length, input_dims] -> [batch, length, emb_dim]
    x = layers.DenseGeneral(
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        kernel_axes=('vocab', 'embed'),
        name='input_proj')(encoder_inputs)

    if cfg.context_positions == 'regular':
      input_positions = jnp.arange(max_positions)[None, :]
    elif cfg.context_positions == 'terminal_relative':
      input_positions = jnp.broadcast_to(
          jnp.arange(max_positions), encoder_inputs.shape[:2])
      seq_lens = jax.vmap(get_sequence_length)(encoder_inputs_mask)
      input_positions = jax.vmap(make_sequence_terminal_relative)(
          input_positions, seq_lens)
    else:
      raise ValueError(f'Unknown context_positions: {cfg.context_positions}')

    x += position_encoding_layer(config=cfg, max_length=max_positions)(
        input_positions)

    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
          config=cfg,
          name=f'layers_{lyr}')(
              inputs=x,
              encoder_inputs_mask=encoder_inputs_mask,
              deterministic=deterministic)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)
    return x, encoder_inputs_mask


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config

  @nn.compact
  def __call__(self,
               encodings_and_masks: jnp.ndarray,
               decoder_input_tokens: jnp.ndarray,
               decoder_noise_time: jnp.ndarray,
               deterministic: bool):
    cfg = self.config
    batch, _, _ = decoder_input_tokens.shape

    assert decoder_noise_time.shape == (batch,)

    # decoder_noise_time is in [0, 1), so rescale to expected timing range.

    conditioning_emb = diffusion_utils.get_timing_signal_1d(
        decoder_noise_time * cfg.max_decoder_noise_time, cfg.emb_dim,
        max_timescale=cfg.max_decoder_noise_time)
    conditioning_emb = layers.DenseGeneral(
        features=cfg.emb_dim * 4,
        dtype=cfg.dtype,
        kernel_axes=('vocab', 'embed'),
        name='time_emb_dense0')(conditioning_emb)
    conditioning_emb = nn.swish(conditioning_emb)
    conditioning_emb = layers.DenseGeneral(
        features=cfg.emb_dim * 4,
        dtype=cfg.dtype,
        kernel_axes=('vocab', 'embed'),
        name='time_emb_dense1')(conditioning_emb)
    conditioning_emb = nn.swish(conditioning_emb)
    conditioning_emb = jnp.expand_dims(conditioning_emb, axis=1)

    assert conditioning_emb.shape == (batch, 1, cfg.emb_dim * 4)

    seq_length = decoder_input_tokens.shape[1]

    # If we want to use relative positions for audio context, we can just offset
    # this sequence by the length of encodings_and_masks.
    decoder_positions = jnp.broadcast_to(
        jnp.arange(seq_length), (batch, seq_length))

    # [batch, length, depth] -> [batch, length, emb_dim]
    assert decoder_input_tokens.ndim == 3

    position_encodings = position_encoding_layer(
        config=cfg, max_length=seq_length)(decoder_positions)

    # decoder: No padding present.
    decoder_mask = jnp.ones(decoder_input_tokens.shape[:2])

    # Translate encoding masks to encoder-decoder masks.
    def encoder_decoder_mask(encoder_mask):
      return layers.make_attention_mask(
          decoder_mask, encoder_mask, dtype=cfg.dtype)

    encodings_and_encdec_masks = [
        (x, encoder_decoder_mask(y)) for x, y in encodings_and_masks]

    inputs = layers.DenseGeneral(
        cfg.emb_dim,
        dtype=cfg.dtype,
        kernel_init=nn.linear.default_kernel_init,
        kernel_axes=('vocab', 'embed'),
        name='continuous_inputs_projection')(decoder_input_tokens)

    inputs += position_encodings

    inputs = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            inputs, deterministic=deterministic)
    inputs = inputs.astype(cfg.dtype)

    y = inputs

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = DecoderLayer(
          config=cfg, name=f'layers_{lyr}')(
              y,
              encodings_and_encdec_masks,
              conditioning_emb=conditioning_emb,
              deterministic=deterministic)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, depth]
    n_out = decoder_input_tokens.shape[-1]
    spec_out = layers.DenseGeneral(
        n_out,
        dtype=jnp.float32,  # Use float32 for stabiliity.
        kernel_axes=('embed', 'vocab'),
        name='spec_out_dense')(y)
    return spec_out


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config

  def setup(self):
    cfg = self.config

    self.encoder = TokenEncoder(config=cfg)
    self.decoder = Decoder(config=cfg)

  def encode(self,
             encoder_input_tokens,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    assert encoder_input_tokens.ndim == 2  # (batch, length)

    encoder_inputs_mask = encoder_input_tokens > 0

    encoded, encoder_inputs_mask = self.encoder(
        encoder_input_tokens=encoder_input_tokens,
        encoder_inputs_mask=encoder_inputs_mask,
        deterministic=not enable_dropout)
    return [(encoded, encoder_inputs_mask)]

  def decode(
      self,
      encodings_and_masks,
      decoder_input_tokens,
      decoder_noise_time,
      enable_dropout=True):
    """Applies Transformer decoder-branch on encoded-input and target."""
    logits = self.decoder(
        encodings_and_masks=encodings_and_masks,
        decoder_input_tokens=decoder_input_tokens,
        decoder_noise_time=decoder_noise_time,
        deterministic=not enable_dropout)
    return logits.astype(self.config.dtype)

  def __call__(self,
               encoder_input_tokens,
               decoder_input_tokens,
               decoder_noise_time,
               *,
               enable_dropout: bool = True):
    """Applies Transformer model on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_noise_time: noise continuous time for diffusion.
      enable_dropout: Ensables dropout if set to True.

    Returns:
      logits array from full transformer.
    """
    encodings_and_masks = self.encode(
        encoder_input_tokens=encoder_input_tokens,
        enable_dropout=enable_dropout)

    return self.decode(
        encodings_and_masks=encodings_and_masks,
        decoder_input_tokens=decoder_input_tokens,
        decoder_noise_time=decoder_noise_time,
        enable_dropout=enable_dropout)


class ContinuousContextTransformer(nn.Module):
  """An encoder-decoder Transformer model with a second audio context encoder."""
  config: T5Config

  def setup(self):
    cfg = self.config

    self.token_encoder = TokenEncoder(config=cfg)
    self.continuous_encoder = ContinuousEncoder(config=cfg)
    self.decoder = Decoder(config=cfg)

  def encode(self,
             input_tokens,
             continuous_inputs,
             continuous_mask,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    assert input_tokens.ndim == 2  # (batch, length)
    assert continuous_inputs.ndim == 3  # (batch, length, input_dims)

    tokens_mask = input_tokens > 0

    tokens_encoded, tokens_mask = self.token_encoder(
        encoder_input_tokens=input_tokens,
        encoder_inputs_mask=tokens_mask,
        deterministic=not enable_dropout)

    continuous_encoded, continuous_mask = self.continuous_encoder(
        encoder_inputs=continuous_inputs,
        encoder_inputs_mask=continuous_mask,
        deterministic=not enable_dropout)

    return [(tokens_encoded, tokens_mask),
            (continuous_encoded, continuous_mask)]

  def decode(
      self,
      encodings_and_masks,
      input_tokens,
      noise_time,
      enable_dropout=True):
    """Applies Transformer decoder-branch on encoded-input and target."""
    logits = self.decoder(
        encodings_and_masks=encodings_and_masks,
        decoder_input_tokens=input_tokens,
        decoder_noise_time=noise_time,
        deterministic=not enable_dropout)
    return logits.astype(self.config.dtype)

  def __call__(self,
               encoder_input_tokens,
               encoder_continuous_inputs,
               encoder_continuous_mask,
               decoder_input_tokens,
               decoder_noise_time,
               *,
               enable_dropout: bool = True):
    """Applies Transformer model on the inputs.

    Args:
      encoder_input_tokens: input data to the encoder.
      encoder_continuous_inputs: continuous inputs for the second encoder.
      encoder_continuous_mask: mask for continuous inputs.
      decoder_input_tokens: input token to the decoder.
      decoder_noise_time: noise continuous time for diffusion.
      enable_dropout: Ensables dropout if set to True.

    Returns:
      logits array from full transformer.
    """
    encodings_and_masks = self.encode(
        input_tokens=encoder_input_tokens,
        continuous_inputs=encoder_continuous_inputs,
        continuous_mask=encoder_continuous_mask,
        enable_dropout=enable_dropout)

    return self.decode(
        encodings_and_masks=encodings_and_masks,
        input_tokens=decoder_input_tokens,
        noise_time=decoder_noise_time,
        enable_dropout=enable_dropout)
