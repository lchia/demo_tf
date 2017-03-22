# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

TEST_PERIOD = 100

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  training_accuracy_summary = tf.summary.scalar(
      "training_accuracy", accuracy)
  validation_accuracy_summary = tf.summary.scalar(
      "validation_accuracy", accuracy)

  training_loss_summary = tf.summary.scalar(
      "training_loss", cross_entropy)
  validation_loss_summary = tf.summary.scalar(
      "validation_loss", cross_entropy)

  sess = tf.InteractiveSession()
  summary_writer = tf.summary.FileWriter(FLAGS.run_dir, sess.graph)

  tf.global_variables_initializer().run()
  # Train
  for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 
    if step % TEST_PERIOD == 0:
      # Training Accuracy
      train_acc, train_acc_summ = sess.run(
          [accuracy, training_accuracy_summary],
          feed_dict={x: mnist.train.images,
                     y_: mnist.train.labels})
      print('Train Accuracy: ', train_acc)
      summary_writer.add_summary(train_acc_summ, step)
      # Validation Accuracy
      validation_acc, validation_acc_summ = sess.run(
          [accuracy, validation_accuracy_summary],
          feed_dict={x: mnist.test.images,
                     y_: mnist.test.labels})
      print('Validation Accuracy: ', validation_acc)
      summary_writer.add_summary(validation_acc_summ, step)
  
      # Training Loss
      train_loss, train_loss_summ = sess.run(
            [cross_entropy, training_loss_summary],
            feed_dict={x: mnist.test.images,
                       y_: mnist.test.labels})
      print('Train Loss: ', train_loss)
      summary_writer.add_summary(train_loss_summ, step)
      # Validation Loss
      validation_loss, validation_loss_summ = sess.run(
            [cross_entropy, validation_loss_summary],
            feed_dict={x: mnist.test.images,
                       y_: mnist.test.labels})
      print('validation Loss: ', validation_loss)
      summary_writer.add_summary(validation_loss_summ, step)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--run_dir', type=str, default='/tmp/tensorflow/mnist/training',
                      help='Directory for storing logs data')
  FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
