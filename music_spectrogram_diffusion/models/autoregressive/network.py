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
import jax.numpy as jnp
from music_spectrogram_diffusion import layers


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
  output_dim: int = 0  # Defaults to natural output size.
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False


class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: T5Config

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, deterministic=False):
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
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
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(dtype=cfg.dtype,
                         name='pre_self_attention_layer_norm')(inputs)

    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='self_attention')(
            x,
            x,
            decoder_mask,
            deterministic=deterministic,
            decode=decode)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # Encoder-Decoder block.
    y = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
            x)
    y = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='encoder_decoder_attention')(
            y, encoded, encoder_decoder_mask, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
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


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config

  @nn.compact
  def __call__(self,
               encoder_input_tokens,
               encoder_mask=None,
               deterministic=False):
    cfg = self.config

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
    x = x + layers.FixedEmbed(features=cfg.emb_dim)(inputs_positions)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
          config=cfg,
          name=f'layers_{lyr}')(x, encoder_mask, deterministic)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config

  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config

    seq_length = decoder_input_tokens.shape[1]
    decoder_positions = jnp.arange(seq_length)[None, :]

    # [batch, length, depth] -> [batch, length, emb_dim]
    assert decoder_input_tokens.ndim == 3
    y = layers.DenseGeneral(
        cfg.emb_dim,
        dtype=cfg.dtype,
        kernel_init=nn.linear.default_kernel_init,
        kernel_axes=('vocab', 'embed'),
        name='continuous_inputs_projection')(decoder_input_tokens)
    y = y + layers.FixedEmbed(features=cfg.emb_dim)(
        decoder_positions, decode=decode)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = DecoderLayer(
          config=cfg, name=f'layers_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              deterministic=deterministic,
              decode=decode,
              max_decode_length=max_decode_length)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, depth]
    n_out = cfg.output_dim or decoder_input_tokens.shape[-1]
    spec_out = layers.DenseGeneral(
        n_out,
        dtype=jnp.float32,  # Use float32 for stabiliity.
        kernel_axes=('embed', 'vocab'),
        name='spec_out_dense')(
            y)
    return spec_out


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config

  def setup(self):
    cfg = self.config

    self.encoder = Encoder(config=cfg)
    self.decoder = Decoder(config=cfg)

  def encode(self,
             encoder_input_tokens,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert encoder_input_tokens.ndim == 2  # (batch, length)

    # Make padding attention mask; we don't actually mask out any input
    # positions, letting the model potentially attend to the zero vector used as
    # padding.
    encoder_mask = layers.make_attention_mask(
        jnp.ones_like(encoder_input_tokens),
        jnp.ones_like(encoder_input_tokens),
        dtype=cfg.dtype)

    return self.encoder(
        encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    # Encoder(notes): >0 ignores padding tokens (padding=0).
    encoder_mask = encoder_input_tokens > 0
    # Target(spectrograms): No padding present.
    target_mask = jnp.ones(decoder_target_tokens.shape[:2])

    # Make padding attention masks.
    encoder_decoder_mask = layers.make_attention_mask(
        target_mask, encoder_mask, dtype=cfg.dtype)

    # fast autoregressive decoding uses only a special encoder-decoder mask
    if decode:
      decoder_mask = None
    else:
      decoder_mask = layers.make_decoder_mask(
          decoder_target_tokens=target_mask, dtype=cfg.dtype)

    logits = self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits.astype(self.config.dtype)

  def __call__(self,
               encoder_input_tokens,
               decoder_input_tokens,
               decoder_target_tokens,
               encoder_segment_ids=None,
               decoder_segment_ids=None,
               encoder_positions=None,
               decoder_positions=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        encoder_input_tokens,
        enable_dropout=enable_dropout)

    return self.decode(
        encoded,
        encoder_input_tokens,  # only used for masks
        decoder_input_tokens,
        decoder_target_tokens,
        decoder_positions=decoder_positions,
        enable_dropout=enable_dropout,
        decode=decode)
