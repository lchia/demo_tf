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

   RUN:
       cd cifar10
       CUDA_VISIBLE_DEVICES=0 python cifar10_train.py 
 
   OUTPUT:
        Downloading cifar-10-binary.tar.gz 100.0%
        Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
        Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
        I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
        ...
        2017-03-22 14:43:15.259168: TEST 0, LOSS = 9.37, TOP1 = 0.0859, TOP5 = 0.4688 (88.8 examples/sec; 1.442 sec/batch)

        2017-03-22 14:43:16.699838: step 10, loss = 6.92, top1 = 0.2188 , top5 = 0.7109 (888.5 examples/sec; 0.144 sec/batch)
        2017-03-22 14:43:18.187086: step 20, loss = 6.81, top1 = 0.1250 , top5 = 0.7031 (860.7 examples/sec; 0.149 sec/batch)

