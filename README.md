# Multi-instrument Music Synthesis with Spectrogram Diffusion

<img src="https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/architecture.png" alt="Architecture diagram">

An ideal music synthesizer should be both interactive and expressive, generating high-fidelity audio in realtime for arbitrary combinations of instruments and notes. Recent neural synthesizers have exhibited a tradeoff between domain-specific models that offer detailed control of only specific instruments, or raw waveform models that can train on any music but with minimal control and slow generation. In this work, we focus on a middle ground of neural synthesizers that can generate audio from MIDI sequences with arbitrary combinations of instruments in realtime. This enables training on a wide range of transcription datasets with a single model, which in turn offers note-level control of composition and instrumentation across a wide range of instruments.  We use a simple two-stage process: MIDI to spectrograms with an encoder-decoder Transformer, then spectrograms to audio with a generative adversarial network (GAN) spectrogram inverter. We compare training the decoder as an autoregressive model and as a Denoising Diffusion Probabilistic Model (DDPM) and find that the DDPM approach is superior both qualitatively and as measured by audio reconstruction and Fréchet distance metrics. Given the interactivity and generality of this approach, we find this to be a promising first step towards interactive and expressive neural synthesis for arbitrary combinations of instruments and notes.


For more information:

* [ISMIR 2022 paper](https://arxiv.org/abs/2206.05408) on arXiv
* [Online Supplement](https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/index.html) with audio examples

## Demo

Upload your own MIDI file and synthesize it with our model:
[Colab Demo](https://colab.research.google.com/github/magenta/music-spectrogram-diffusion/blob/main/music_spectrogram_diffusion/colab/synthesize_midi.ipynb)

Note that the TPUs provided by Colab are significantly slower than the ones used in the paper, so the `base_with_context` model will render about 5x slower than realtime.

For results similar to the speeds described in the paper, a TPUv4 accelerator is required.

## Pretrained Models

* [base_with_context](https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/checkpoints/base_with_context.zip) (best model)
* [small with context](https://storage.googleapis.com/music-synthesis-with-spectrogram-diffusion/checkpoints/small_with_context.zip)
* [SoundStream spectrogram inverter](https://tfhub.dev/google/soundstream/mel/decoder/music/1)

## Disclaimer

This is not an officially supported Google product.
