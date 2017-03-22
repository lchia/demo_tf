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

"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    test_images, test_labels = cifar10.inputs(True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    with tf.variable_scope("Inference"):
      logits = cifar10.inference(images)
    with tf.variable_scope("Inference", reuse=True):
      test_logits = cifar10.inference(test_images)

    # Calculate loss.
    with tf.name_scope("train_validation_loss"):
      loss, top1, top5 = cifar10.loss(logits, labels)

      test_loss, test_top1, test_top5 = cifar10.loss(test_logits, test_labels)
      tf.summary.scalar('test_loss', test_loss)
      tf.summary.scalar('train_loss', loss)
      tf.summary.scalar('test_top1', test_top1)
      tf.summary.scalar('train_top1', top1)
      tf.summary.scalar('test_top5', test_top5)
      tf.summary.scalar('train_top5', top5)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1 
        if self._step % 100 ==0:
          return tf.train.SessionRunArgs([test_loss, test_top1, test_top5])
        else: 
          return tf.train.SessionRunArgs([loss, top1, top5]) # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results[0]
          top1_value = run_values.results[1]
          top5_value = run_values.results[2] 
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          if self._step % 100 ==0:
            format_str = ('%s: TEST %d, LOSS = %.2f, TOP1 = %.4f, TOP5 = %.4f '
                          '(%.1f examples/sec; %.3f sec/batch)\n')
            print (format_str % (datetime.now(), self._step, loss_value,
                                 top1_value, top5_value, 
                                 examples_per_sec, sec_per_batch))
          else:
            format_str = ('%s: step %d, loss = %.2f, top1 = %.4f , top5 = %.4f '
                          '(%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value,
                                 top1_value, top5_value, 
                                 examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
