# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of util functions for model construction.
"""
import numpy
import tensorflow as tf
from tensorflow import logging
from tensorflow import flags
import tensorflow.contrib.slim as slim

def get_avg_pooled(model_input, num_frames):
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    return tf.reduce_sum(model_input,
                            axis=[1]) / denominators

def SampleRandomSequence(model_input, num_frames, num_samples):
  """Samples a random sequence of frames of size num_samples.

  Args:
    model_input: A tensor of size batch_size x max_frames x feature_size
    num_frames: A tensor of size batch_size x 1
    num_samples: A scalar

  Returns:
    `model_input`: A tensor of size batch_size x num_samples x feature_size
  """

  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)


def SampleRandomFrames(model_input, num_frames, num_samples):
  """Samples a random set of frames of size num_samples.

  Args:
    model_input: A tensor of size batch_size x max_frames x feature_size
    num_frames: A tensor of size batch_size x 1
    num_samples: A scalar

  Returns:
    `model_input`: A tensor of size batch_size x num_samples x feature_size
  """
  batch_size = tf.shape(model_input)[0]
  frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, num_samples]),
          tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)

def FramePooling(frames, method, **unused_params):
  """Pools over the frames of a video.

  Args:
    frames: A tensor with shape [batch_size, num_frames, feature_size].
    method: "average", "max", "attention", or "none".
  Returns:
    A tensor with shape [batch_size, feature_size] for average, max, or
    attention pooling. A tensor with shape [batch_size*num_frames, feature_size]
    for none pooling.

  Raises:
    ValueError: if method is other than "average", "max", "attention", or
    "none".
  """
  if method == "average":
    return tf.reduce_mean(frames, 1)
  elif method == "max":
    return tf.reduce_max(frames, 1)
  elif method == "none":
    feature_size = frames.shape_as_list()[2]
    return tf.reshape(frames, [-1, feature_size])
  else:
    raise ValueError("Unrecognized pooling method: %s" % method)

def make_fully_connected_net(input_, sizes, vocab_size,
    l2_penalty, batch_norm=False):
    layers = []
    normalizer = slim.batch_norm if batch_norm else None
    for size in sizes:
        prev = layers[-1] if len(layers) else input_
        layers.append(
            slim.fully_connected(
                prev,
                size,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                normalizer_fn=normalizer
            )
        )
    return slim.fully_connected(
        layers[-1] if len(layers) else input_,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty)
    )

def make_conv_relu_pool(input_, conv_size, pool_size, num_channels,
    batch_norm=False):
    input_shape = input_.get_shape().as_list()
    normalizer = slim.batch_norm if batch_norm else None
    conv = slim.conv2d(input_, num_channels, [conv_size, 1],
        normalizer_fn=normalizer)
    return tf.nn.max_pool(
        conv, [1, pool_size, 1, 1], [1, pool_size, 1, 1], 'SAME')
