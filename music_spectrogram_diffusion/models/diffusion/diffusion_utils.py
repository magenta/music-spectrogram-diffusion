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

"""Diffusion utilities."""

from typing import Callable, Mapping, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np


@flax.struct.dataclass
class DiffusionSchedule:
  name: str
  start: Optional[float] = None
  stop: Optional[float] = None
  num_steps: Optional[int] = None


@flax.struct.dataclass
class ClassifierFreeGuidanceConfig:
  drop_condition_prob: float = 0.1
  eval_condition_weight: float = 5.0


@flax.struct.dataclass
class SamplerConfig:
  name: str = "ddpm"
  schedule: DiffusionSchedule = DiffusionSchedule(
      name="cosine",
      num_steps=1000)
  clip_x0: bool = True
  logvar_type: str = "large"


@flax.struct.dataclass
class DiffusionConfig:
  """Diffusion parameters."""
  time_continuous_or_discrete: str = "continuous"
  train_schedule: DiffusionSchedule = DiffusionSchedule(name="cosine")
  loss_norm: str = "l1"
  loss_type: str = "eps"
  model_output: str = "eps"
  classifier_free_guidance: ClassifierFreeGuidanceConfig = (
      ClassifierFreeGuidanceConfig())
  sampler: SamplerConfig = SamplerConfig()


def broadcast_to_shape_from_left(
    x: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
  assert len(shape) >= x.ndim
  return jnp.broadcast_to(
      x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)


def get_timing_signal_1d(position: jnp.ndarray,
                         num_channels: int,
                         min_timescale: float = 1.0,
                         max_timescale: float = 2.0e4) -> jnp.ndarray:
  """Returns the positional encoding (same as Tensor2Tensor).

  Args:
    position: An array of shape [batch_size].
    num_channels: The number of output channels.
    min_timescale: The smallest time unit (should probably be 0.0).
    max_timescale: The largest time unit.

  Returns:
    a Tensor of timing signals [1, length, num_channels]
  """
  assert position.ndim == 1
  assert num_channels % 2 == 0
  num_timescales = float(num_channels // 2)
  log_timescale_increment = (
      np.log(max_timescale / min_timescale) / (num_timescales - 1.0))
  inv_timescales = min_timescale * jnp.exp(
      jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
  scaled_time = (jnp.expand_dims(position, 1) *
                 jnp.expand_dims(inv_timescales, 0))
  # Please note that this slightly differs from the published paper.
  # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
  signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)
  signal = jnp.reshape(signal, [jnp.shape(position)[0], num_channels])
  return signal


def log1mexp(x: jnp.ndarray) -> jnp.ndarray:
  """Accurate computation of log(1 - exp(-x)) for x > 0."""
  # From James Townsend's PixelCNN++ code
  # Method from
  # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
  return jnp.where(x > jnp.log(2), jnp.log1p(-jnp.exp(-x)),
                   jnp.log(-jnp.expm1(-x)))


def diffusion_forward(
    *, x0: jnp.ndarray, logsnr: jnp.ndarray) -> Mapping[str, jnp.ndarray]:
  """Forward diffusion process: q(z_t | x0)."""
  return {
      "mean": x0 * jnp.sqrt(jax.nn.sigmoid(logsnr)),
      "std": jnp.sqrt(jax.nn.sigmoid(-logsnr)),
      "var": jax.nn.sigmoid(-logsnr),
      "logvar": jax.nn.log_sigmoid(-logsnr),
  }


def diffusion_reverse(
    *,
    x0: jnp.ndarray,
    z_t: jnp.ndarray,
    logsnr_s: jnp.ndarray,
    logsnr_t: jnp.ndarray,
    logvar_type: str) -> Mapping[str, jnp.ndarray]:
  """Reverse diffusion process: q(z_s | z_t, x0)."""

  # Requires: logsnr_s > logsnr_t (i.e., s < t).

  alpha_st = jnp.sqrt((1. + jnp.exp(-logsnr_t)) / (1. + jnp.exp(-logsnr_s)))
  alpha_s = jnp.sqrt(jax.nn.sigmoid(logsnr_s))

  r = jnp.exp(logsnr_t - logsnr_s)  # SNR(t) / SNR(s)
  one_minus_r = -jnp.expm1(logsnr_t - logsnr_s)  # 1 - SNR(t) / SNR(s)
  log_one_minus_r = log1mexp(logsnr_s - logsnr_t)  # log(1 - SNR(t) / SNR(s))

  mean = r * alpha_st * z_t + one_minus_r * alpha_s * x0

  # Currently, we only support fixed log-variance types.
  if logvar_type == "small":
    # same as setting x_logvar to -infinity
    var = one_minus_r * jax.nn.sigmoid(-logsnr_s)
    logvar = log_one_minus_r + jax.nn.log_sigmoid(-logsnr_s)
  elif logvar_type == "large":
    # same as setting x_logvar to nn.log_sigmoid(-logsnr_t)
    var = one_minus_r * jax.nn.sigmoid(-logsnr_t)
    logvar = log_one_minus_r + jax.nn.log_sigmoid(-logsnr_t)
  elif logvar_type.startswith("medium:"):
    _, frac = logvar_type.split(":")
    frac = float(frac)
    assert 0 <= frac <= 1
    min_logvar = log_one_minus_r + jax.nn.log_sigmoid(-logsnr_s)
    max_logvar = log_one_minus_r + jax.nn.log_sigmoid(-logsnr_t)
    logvar = frac * max_logvar + (1 - frac) * min_logvar
    var = jnp.exp(logvar)

  return {
      "mean": mean,
      "std": jnp.sqrt(var),
      "var": var,
      "logvar": logvar,
  }


def get_logsnr_t(t: jnp.ndarray, schedule: DiffusionSchedule) -> jnp.ndarray:
  """Return log-SNR at a given t for a given schedule.

  Args:
    t: The continuous time between [0.0, 1.0].
    schedule: The noise schedule.

  Returns:
    The log signal-to-noise ratio, in the current implementation
    logsnr_min == -20.0, and logsnr_max == 20.0.
  """
  # The range of logsnr value is clipped between [-20.0, 20.0]
  logsnr_min = -20.0
  logsnr_max = 20.0

  if schedule.name == "cosine":
    # This is based on the schedule alpha_cumprod_t = cos(t * pi / 2) ** 2.
    # Note that OpenAI cosine schedule is slightly different from this, where
    # alpha_cumprod_t = cos(((t + 0.008) / 1.008) * pi / 2) ** 2.
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    return -2.0 * jnp.log(jnp.tan(a * t + b))

  elif schedule.name == "linear":
    # A linear noise schedule where beta is linearly interpolated from
    # schedule.start to schedule.stop.
    assert schedule.num_steps > 0
    betas = np.linspace(
        schedule.start, schedule.stop, schedule.num_steps, dtype=np.float64)
    alphas_cumprod = np.cumprod(1. - betas, axis=0)
    logsnr = np.log(alphas_cumprod) - np.log1p(-alphas_cumprod)
    # Clip the values between [-logsnr_min, logsnr_max]
    logsnr = np.clip(logsnr, a_min=logsnr_min, a_max=logsnr_max)
    return jnp.interp(t, np.linspace(0, 1, schedule.num_steps), logsnr)

  else:
    raise ValueError("Schedule %s not identified." % schedule.name)


def predict_eps_from_x0(*,
                        z: jnp.ndarray,
                        x0: jnp.ndarray,
                        logsnr: jnp.ndarray) -> jnp.ndarray:
  """eps = (z - alpha * x0) / sigma."""
  logsnr = broadcast_to_shape_from_left(logsnr, z.shape)
  return jnp.sqrt(1.0 + jnp.exp(logsnr)) * (
      z - x0 * jax.lax.rsqrt(1.0 + jnp.exp(-logsnr)))


def predict_x0_from_eps(*,
                        z: jnp.ndarray,
                        eps: jnp.ndarray,
                        logsnr: jnp.ndarray) -> jnp.ndarray:
  """x0 = (z - sigma * eps) / alpha."""
  logsnr = broadcast_to_shape_from_left(logsnr, z.shape)
  return jnp.sqrt(1.0 + jnp.exp(-logsnr)) * (
      z - eps * jax.lax.rsqrt(1.0 + jnp.exp(logsnr)))


def predict_x0_from_v(*,
                      z: jnp.ndarray,
                      v: jnp.ndarray,
                      logsnr: jnp.ndarray) -> jnp.ndarray:
  """x0 = alpha * z - sigma * v."""
  logsnr = broadcast_to_shape_from_left(logsnr, z.shape)
  alpha_t = jnp.sqrt(jax.nn.sigmoid(logsnr))
  sigma_t = jnp.sqrt(jax.nn.sigmoid(-logsnr))
  return alpha_t * z - sigma_t * v


def get_diffusion_training_input(
    rng: jax.Array,
    x0: jnp.ndarray,
    diffusion_config: DiffusionConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Get model input for training.

  Args:
    rng: RNG key.
    x0: Original data.
    diffusion_config: Diffusion config object.

  Returns:
    z_t: Data after forward diffusion.
    eps: Sampled noise used.
    time: Sampled time used in range [0, 1).
    include_conditioning: Whether conditioning information should be passed to
      the model.
  """
  rng_eps, rng_t, rng_cond = jax.random.split(rng, 3)
  del rng

  eps = jax.random.normal(rng_eps, shape=x0.shape)

  batch_size = x0.shape[0]
  # Sample time.
  if diffusion_config.time_continuous_or_discrete == "continuous":
    time = jax.random.uniform(rng_t, (batch_size,))
  elif diffusion_config.time_continuous_or_discrete == "discrete":
    time = jax.random.randint(
        rng_t, (batch_size,), 0, diffusion_config.train_schedule.num_steps)
    # TODO(watsondaniel): divide by T - 1??
    time = (time.astype(jnp.float32) /
            float(diffusion_config.train_schedule.num_steps))
  else:
    raise ValueError("Invalid value for time_continuous_or_discrete: %s" %
                     diffusion_config.time_continuous_or_discrete)

  logsnr = get_logsnr_t(time, diffusion_config.train_schedule)

  # Sample z ~ q(z_t | x0).
  z_dist = diffusion_forward(
      x0=x0, logsnr=broadcast_to_shape_from_left(logsnr, x0.shape))
  z_t = z_dist["mean"] + z_dist["std"] * eps

  include_conditioning = jax.random.bernoulli(
      rng_cond, shape=[x0.shape[0]],
      p=1 - diffusion_config.classifier_free_guidance.drop_condition_prob)

  return z_t, eps, time, include_conditioning


def _get_x0_and_eps_from_model_output(
    z: jnp.ndarray,
    time: jnp.ndarray,
    model_output: jnp.ndarray,
    diffusion_config: DiffusionConfig) -> Mapping[str, jnp.ndarray]:
  """Get x0 and eps from model output, converting as necessary."""
  logsnr = get_logsnr_t(time, diffusion_config.train_schedule)

  if diffusion_config.model_output == "eps":
    outputs = {
        "eps": model_output,
        "x0": predict_x0_from_eps(z=z, eps=model_output, logsnr=logsnr),
    }
  elif diffusion_config.model_output == "x0":
    outputs = {
        "eps": predict_eps_from_x0(z=z, x0=model_output, logsnr=logsnr),
        "x0": model_output,
    }
  elif diffusion_config.model_output == "x0_and_eps":
    x0_, eps_ = jnp.split(model_output, 2, axis=-1)
    x0 = predict_x0_from_eps(z=z, eps=eps_, logsnr=logsnr)
    wx = broadcast_to_shape_from_left(jax.nn.sigmoid(-logsnr), z.shape)
    x0_out = wx * x0_ + (1. - wx) * x0
    eps_out = predict_eps_from_x0(z=z, x0=x0_out, logsnr=logsnr)
    outputs = {"x0": x0_out, "eps": eps_out}
  elif diffusion_config.model_output == "v":
    x0_out = predict_x0_from_v(z=z, v=model_output, logsnr=logsnr)
    outputs = {
        "x0": x0_out,
        "eps": predict_eps_from_x0(z=z, x0=x0_out, logsnr=logsnr)
    }
  else:
    raise ValueError("Unknown model_output: %s" % diffusion_config.model_output)

  return outputs


def calculate_loss(
    x0: jnp.ndarray,
    eps: jnp.ndarray,
    z: jnp.ndarray,
    time: jnp.ndarray,
    model_output: jnp.ndarray,
    diffusion_config: DiffusionConfig) -> jnp.ndarray:
  """Calculate diffusion loss."""
  outputs = _get_x0_and_eps_from_model_output(
      z=z,
      time=time,
      model_output=model_output,
      diffusion_config=diffusion_config)

  def diffusion_loss(a, b):
    """L1 or L2."""
    if diffusion_config.loss_norm == "l1":
      loss = jnp.abs(a - b)
    elif diffusion_config.loss_norm == "l2":
      loss = jnp.square(a - b)
    else:
      raise ValueError("Unknown diffusion loss norm: %s" %
                       diffusion_config.loss_norm)

    return loss

  x0_loss = diffusion_loss(outputs["x0"], x0)
  eps_loss = diffusion_loss(outputs["eps"], eps)

  if diffusion_config.loss_type == "x0":
    loss = x0_loss
  elif diffusion_config.loss_type == "eps":
    loss = eps_loss
  elif diffusion_config.loss_type == "max_x0_eps":
    loss = jnp.maximum(x0_loss, eps_loss)
  elif diffusion_config.loss_type == "x0_and_eps":
    loss = eps_loss + x0_loss
  else:
    raise ValueError("Unknown diffusion loss_type: %s" %
                     diffusion_config.loss_type)

  return loss


def ddim_step(i: jnp.ndarray, logsnr_s: jnp.ndarray,
              logsnr_t: jnp.ndarray, pred_x_t: jnp.ndarray,
              pred_eps_t: jnp.ndarray) -> jnp.ndarray:
  """Sample one step of DDIM."""
  del logsnr_t
  logsnr_s = broadcast_to_shape_from_left(logsnr_s, pred_x_t.shape)

  stdv_s = jnp.sqrt(jax.nn.sigmoid(-logsnr_s))
  alpha_s = jnp.sqrt(jax.nn.sigmoid(logsnr_s))
  z_s_pred = alpha_s * pred_x_t + stdv_s * pred_eps_t
  return jnp.where(i == 0, pred_x_t, z_s_pred)


def ddpm_step(i: jnp.ndarray, rng: jnp.ndarray, logsnr_s: jnp.ndarray,
              logsnr_t: jnp.ndarray, pred_x0: jnp.ndarray,
              z_t: jnp.ndarray, logvar_type: str) -> jnp.ndarray:
  """Sample one step of DDPM."""
  logsnr_s = broadcast_to_shape_from_left(logsnr_s, pred_x0.shape)
  logsnr_t = broadcast_to_shape_from_left(logsnr_t, pred_x0.shape)

  eps = jax.random.normal(
      jax.random.fold_in(rng, i), shape=pred_x0.shape, dtype=pred_x0.dtype)

  z_s_dist = diffusion_reverse(
      x0=pred_x0, z_t=z_t, logsnr_s=logsnr_s, logsnr_t=logsnr_t,
      logvar_type=logvar_type)
  return jnp.where(i == 0, pred_x0, z_s_dist["mean"] + z_s_dist["std"] * eps)


def eval_step(
    rng: jax.Array,
    diffusion_config: DiffusionConfig,
    batch_size: int,
    pred_fn: Callable[..., jnp.ndarray]
) -> Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, None]]:
  """Create a function that does one diffusion reverse step."""
  schedule = diffusion_config.sampler.schedule
  num_steps = schedule.num_steps

  def body(z_t, i):
    t = (i + 1.0).astype(jnp.float32) / num_steps
    s = i.astype(jnp.float32) / num_steps
    logsnr_t = jnp.full((batch_size,), get_logsnr_t(t, schedule))
    logsnr_s = jnp.full((batch_size,), get_logsnr_t(s, schedule))
    time = jnp.full((batch_size,), t)

    model_output = pred_fn(z=z_t, time=time, include_conditioning=True)
    outputs = _get_x0_and_eps_from_model_output(
        z=z_t,
        time=time,
        model_output=model_output,
        diffusion_config=diffusion_config)
    pred_eps = outputs["eps"]
    pred_x0 = outputs["x0"]

    if diffusion_config.classifier_free_guidance.eval_condition_weight != 1:
      # Classifier-Free Diffusion Guidance, Ho and Salimans 2021
      cond_wt = diffusion_config.classifier_free_guidance.eval_condition_weight
      uncond_wt = 1. - cond_wt
      uncond_model_output = pred_fn(
          z=z_t, time=time, include_conditioning=False)
      uncond_outputs = _get_x0_and_eps_from_model_output(
          z=z_t,
          time=time,
          model_output=uncond_model_output,
          diffusion_config=diffusion_config)
      pred_eps = cond_wt * pred_eps + uncond_wt * uncond_outputs["eps"]
      pred_x0 = predict_x0_from_eps(z=z_t, eps=pred_eps, logsnr=logsnr_t)

    # TODO(williamchan): Modify the sampler according to the mask.

    if diffusion_config.sampler.clip_x0:
      pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)
      pred_eps = predict_eps_from_x0(z=z_t, x0=pred_x0, logsnr=logsnr_t)

    if diffusion_config.sampler.name == "ddim":
      z_t = ddim_step(i, logsnr_s, logsnr_t, pred_x0, pred_eps)
    elif diffusion_config.sampler.name == "ddpm":
      z_t = ddpm_step(i, rng, logsnr_s, logsnr_t, pred_x0, z_t,
                      diffusion_config.sampler.logvar_type)
    else:
      raise ValueError("Unknown sampler type: %s" %
                       diffusion_config.sampler.name)
    return z_t, None
  return body


def eval_scan(rng: jax.Array,
              target_shape: Tuple[int],
              pred_fn: Callable[..., jnp.ndarray],
              diffusion_config: DiffusionConfig) -> jnp.ndarray:
  """Scan through all reverse diffusion steps for a batch."""
  # TODO(sahariac): Making this bfloat16 led to much worse results. Why?
  init_z = jax.random.normal(rng, shape=target_shape, dtype=jnp.float32)

  batch_size = init_z.shape[0]

  step_fn = eval_step(
      rng=rng, diffusion_config=diffusion_config,
      batch_size=batch_size, pred_fn=pred_fn)

  pred_x0, _ = jax.lax.scan(
      f=step_fn,
      init=init_z,
      xs=jnp.arange(0, diffusion_config.sampler.schedule.num_steps),
      reverse=True)

  return pred_x0
