1. mnist/mnist_softmax.py
   https://www.tensorflow.org/get_started/mnist/beginners
   0.9134
   
   Demo of the simplest tf training: 
       data loading
       training
       testing
       summary

2. mnist/mnist_with_summaries.py
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
   0.9664

   Demo of summaries with super clear structures, just see from tensorboard.

     cd /tmp/tensorflow/mnist/logs/mnist_with_summaries/train
     tensorboard --logdit=.
   
   with train/val loss together, image summary, distributions, histograms.

3. cifar10/
   https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
   cifar10 with train-validation-loss-accuracy
