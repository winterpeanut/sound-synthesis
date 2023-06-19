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

"""Defines routines to compute mel spectrogram features from audio waveform."""

import numpy as np
import tensorflow as tf

def frame(data, window_length, hop_length):
    
    num_samples = tf.shape(data)[0]
    num_frames = 1 + tf.cast(tf.floor((num_samples - window_length) / hop_length), tf.int32)
    shape = tf.concat( [[num_frames, window_length],data.shape[1:]], axis=0)
    frames = tf.TensorArray(dtype=data.dtype, size=num_frames)
    for i in tf.range(num_frames):
        start = i * hop_length
        end = start + window_length
        frames = frames.write(i, tf.expand_dims(data[start:end], axis=0))
    frames = tf.reshape(frames.stack(), shape)
    return frames# tf.squeeze(frames, axis=-1)


def periodic_hann(window_length):
    tfpi = tf.constant(np.pi,dtype = tf.float64)
    wl = tf.constant(window_length.numpy(),dtype = tf.float64)

    return 0.5 - (0.5 * tf.cos(2 * tfpi / wl *
                             tf.range(wl, dtype=tf.float64)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
    frames = frame(signal, window_length, hop_length)
  
    # Apply frame window to each frame. We use a periodic Hann (cosine of period
    # window_length) instead of the symmetric Hann of np.hanning (period
    # window_length-1).
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    windowed_frames = tf.cast(windowed_frames,tf.float64)
    st = tf.signal.rfft(windowed_frames,[fft_length])
 
    return tf.abs(st)

# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
 
  return _MEL_HIGH_FREQUENCY_Q * tf.math.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

# audio_sample_rate=16000
# log_offset=0.01
# window_length_secs=0.025
# hop_length_secs=0.01
# num_mel_bins=64
# lower_edge_hertz=125
# upper_edge_hertz=7500
def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=257,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
 
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz < 0.0:
    raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > nyquist_hertz:
    raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
  spectrogram_bins_hertz = tf.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
  # The i'th mel band (starting from i=1) has center frequency
  # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
  # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
  # the band_edges_mel arrays.
  band_edges_mel = tf.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  #mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  mel_weights_matrix = tf.TensorArray(tf.float32, size=num_mel_bins)
  for i in tf.range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the *mel* domain, not hertz.
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    
    mel_weights_matrix = mel_weights_matrix.write(i, tf.maximum(0.0, tf.minimum(lower_slope, upper_slope)))
  mel_weights_matrix = tf.transpose(mel_weights_matrix.stack())

# HTK excludes the spectrogram DC bin; make sure it always gets a zero
# coefficient.
  mel_weights_matrix = tf.pad(mel_weights_matrix, [[1, 0], [0, 0]])
  mel_weights_matrix = mel_weights_matrix[:-1]

  return tf.cast(mel_weights_matrix,dtype = tf.float64)
  #return mel_weights_matrix


def log_mel_spectrogram(data,
                        audio_sample_rate=16000,
                        log_offset=0.01,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        num_mel_bins=64,
                        lower_edge_hertz=125,
                        upper_edge_hertz=7500):

  window_length_samples = tf.constant(round(audio_sample_rate * window_length_secs),dtype = tf.int32)
  
  hop_length_samples = tf.constant(round(audio_sample_rate * hop_length_secs),dtype = tf.int32)
  
  fft_length = tf.constant(2 ** int(tf.math.ceil(tf.math.log(tf.cast(window_length_samples,dtype=tf.float32)) / tf.math.log(2.0))),dtype = tf.int32)
  
  
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)
  spectrogram = tf.cast(spectrogram,dtype = tf.float64)

  mel_spectrogram = tf.matmul(spectrogram, spectrogram_to_mel_matrix(
      num_spectrogram_bins=spectrogram.shape[1],
      audio_sample_rate=audio_sample_rate, num_mel_bins=64,lower_edge_hertz=125,upper_edge_hertz=7500))
  return tf.math.log(mel_spectrogram + log_offset)
