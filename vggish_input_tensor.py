# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy

import mel_features_tensor
import vggish_params
import tensorflow as tf

try:
  import soundfile as sf

  def wav_read(wav_file):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    return wav_data, sr

except ImportError:

  def wav_read(wav_file):
    raise NotImplementedError('WAV file reading requires soundfile package.')


def waveform_to_examples(data, sample_rate):
 
  # Convert to mono.
  rank = tf.rank(data)
  if rank > 1:
    data = tf.reduce_mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != 16000:
    data = resampy.resample(data, sample_rate,16000)

  # Compute log mel spectrogram features.
  log_mel = mel_features_tensor.log_mel_spectrogram(
      data,
      audio_sample_rate=16000,
      log_offset=0.01,
      window_length_secs=0.025,
      hop_length_secs=0.01,
      num_mel_bins=64,
      lower_edge_hertz=125,
      upper_edge_hertz=7500)

  # Frame features into examples.
  features_sample_rate = 1.0 / 0.01
  
  # example_window_length = int(round(
  #    0.96 * features_sample_rate))
  
  # example_hop_length = int(round(
  #     0.96 * features_sample_rate))
  
  log_mel_examples = mel_features_tensor.frame(
      log_mel,
      window_length=96,
      hop_length=96)
  
  return log_mel_examples


def wavfile_to_examples(wav_file):
  """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.

  Returns:
    See waveform_to_examples.
  """
  wav_data, sr = wav_read(wav_file)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  return waveform_to_examples(samples, sr)
