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

"""Tests for metrics."""


from absl.testing import absltest
from music_spectrogram_diffusion import metrics
import numpy as np


class MetricsTest(absltest.TestCase):

  def test_streaming_multivariate_gaussian(self):
    smg = metrics.StreamingMultivariateGaussian()
    x = np.random.normal(size=[10, 2])
    smg.update(x[:5])
    self.assertEqual(5, smg.n)
    np.testing.assert_allclose(np.mean(x[:5], axis=0), smg.mu)
    np.testing.assert_allclose(np.cov(x[:5], rowvar=False, ddof=0), smg.sigma)
    smg.update(x[5:])
    self.assertEqual(10, smg.n)
    np.testing.assert_allclose(np.mean(x, axis=0), smg.mu)
    np.testing.assert_allclose(np.cov(x, rowvar=False, ddof=0), smg.sigma)

  def test_streaming_multivariate_gaussian_diagonal_covariance(self):
    smg = metrics.StreamingMultivariateGaussian()
    x = np.random.normal(size=[10, 2048])
    smg.update(x[:5])
    self.assertEqual(5, smg.n)
    np.testing.assert_allclose(np.mean(x[:5], axis=0), smg.mu)
    np.testing.assert_allclose(np.var(x[:5], axis=0, ddof=0), smg.sigma)
    smg.update(x[5:])
    self.assertEqual(10, smg.n)
    np.testing.assert_allclose(np.mean(x, axis=0), smg.mu)
    np.testing.assert_allclose(np.var(x, axis=0, ddof=0), smg.sigma)


if __name__ == '__main__':
  absltest.main()
