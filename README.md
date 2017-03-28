<div align="center">
  <img src="https://github.com/lchia/demo_tf/blob/master/cat.jpeg"><br><br>
</div>
-----------------

1. mnist/mnist_softmax.py

* [mnist website](https://www.tensorflow.org/get_started/mnist/beginners)

   Accuracy: 0.9134
   
   Demo of the simplest tf training:
       data loading
       training
       testing
       summary

2. mnist/mnist_with_summaries.py

* [mnist_summaries website](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)

   Accuracy: 0.9664

   Demo of summaries with super clear structures, just see from tensorboard.

   ```python
   >>> cd /tmp/tensorflow/mnist/logs/mnist_with_summaries/train
   >>> tensorboard --logdit=.
   ```
 
   with train/val loss together, image summary, distributions, histograms.

3. original_cifar10/

* [cifar10 weibsite](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)

   ```python
   >>> cd original_cifar10
   >>>
   >>>  nohup sh single_train_test.sh >single_train_test.out &1 &

   ACCURACY: 

   OUTPUT:

        2017-03-28 19:15:54.410553: step 0, loss = 4.68 (80.0 examples/sec; 1.600 sec/batch)
        2017-03-28 19:15:55.320405: step 10, loss = 4.63 (1406.8 examples/sec; 0.091 sec/batch)
        2017-03-28 19:15:57.503903: step 20, loss = 4.42 (586.2 examples/sec; 0.218 sec/batch)
        2017-03-28 19:15:58.439728: step 30, loss = 4.46 (1367.8 examples/sec; 0.094 sec/batch)
        2017-03-28 19:15:59.277065: step 40, loss = 4.37 (1528.7 examples/sec; 0.084 sec/batch)
        2017-03-28 19:16:00.143080: step 50, loss = 4.31 (1478.0 examples/sec; 0.087 sec/batch)
        ...
        ...
        2017-03-28 19:17:32.317779: precision @ 1 = 0.615


   >>>  nohup sh multi_train_test.sh >multi_train_test.out &1 &

   ACCURACY: 

   OUTPUT:
        2017-03-28 19:20:18.336990: step 0, loss = 4.67 (8.2 examples/sec; 15.559 sec/batch)
        2017-03-28 19:20:19.401855: step 10, loss = 4.62 (1257.0 examples/sec; 0.102 sec/batch)
        2017-03-28 19:20:20.289993: step 20, loss = 4.43 (1584.8 examples/sec; 0.081 sec/batch)
        2017-03-28 19:20:22.698352: step 30, loss = 4.47 (1238.3 examples/sec; 0.103 sec/batch)
        2017-03-28 19:20:23.577287: step 40, loss = 4.29 (1506.1 examples/sec; 0.085 sec/batch)
        2017-03-28 19:20:24.464864: step 50, loss = 4.36 (1547.3 examples/sec; 0.083 sec/batch)
        2017-03-28 19:20:25.327596: step 60, loss = 4.19 (1440.6 examples/sec; 0.089 sec/batch)
        ...
        ...
        2017-03-28 19:22:01.760928: precision @ 1 = 0.605


 

4. my_cifar10/
 
   cifar10 with train-validation-loss-accuracy

   ```python
   >>> cd my_cifar10
   >>>  nohup sh single_train_test.sh >single_train_test.out &1 &

   ACCURACY: 

   OUTPUT:

        Downloading cifar-10-binary.tar.gz 100.0%
        Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
        Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
        I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
        ...
        2017-03-28 19:02:33.235077: step 0, TEST loss = 2.31, TOP_1 = 0.0859, TOP_5 = 0.4922 (64.6 examples/sec; 1.981 sec/batch)
        2017-03-28 19:02:33.954488: step 10, TRAIN loss = 2.28 TOTAL loss = 4.63 (1779.2 examples/sec; 0.072 sec/batch)
        2017-03-28 19:02:34.795683: step 20, TRAIN loss = 2.10 TOTAL loss = 4.44 (1521.6 examples/sec; 0.084 sec/batch)
        2017-03-28 19:02:37.134655: step 30, TRAIN loss = 2.16 TOTAL loss = 4.47 (547.2 examples/sec; 0.234 sec/batch)
        2017-03-28 19:02:38.003000: step 40, TRAIN loss = 2.08 TOTAL loss = 4.38 (1474.1 examples/sec; 0.087 sec/batch)
        2017-03-28 19:02:38.844470: step 50, TRAIN loss = 2.30 TOTAL loss = 4.58 (1521.1 examples/sec; 0.084 sec/batch)
        ...
        2017-03-28 19:02:55.814459: step 200, TEST loss = 1.67, TOP_1 = 0.3828, TOP_5 = 0.8672 (1263.3 examples/sec; 0.101 sec/batch)
        ...


        2017-03-28 19:04:08.756654: precision @ 1 = 0.609


 
   >>>  nohup sh multi_train_test.sh >multi_train_test.out &1 &

   ACCURACY:

   OUTPUT:

        Downloading cifar-10-binary.tar.gz 100.0%
        Successfully downloaded cifar-10-binary.tar.gz 170052171 bytes.
        Filling queue with 20000 CIFAR images before starting to train. This will take a few minutes.
        I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties: 
        ...
        2017-03-28 19:08:26.581715: step 0, loss = 4.67 (8.1 examples/sec; 15.759 sec/batch)
        2017-03-28 19:08:35.062214: TEST 0, loss = 2.28, TOP_1: 0.1159, TOP_5: 0.4937 (8.1 examples/sec; 15.759 sec/batch)
        2017-03-28 19:08:37.534600: step 10, loss = 4.66 (1538.9 examples/sec; 0.083 sec/batch)
        2017-03-28 19:08:38.399939: step 20, loss = 4.56 (1536.0 examples/sec; 0.083 sec/batch)
        2017-03-28 19:08:39.225700: step 30, loss = 4.47 (1549.4 examples/sec; 0.083 sec/batch)
        2017-03-28 19:08:40.077749: step 40, loss = 4.34 (1559.8 examples/sec; 0.082 sec/batch)
        ...
        2017-03-28 19:11:50.167826: TEST 200, loss = 1.67, TOP_1: 0.3856, TOP_5: 0.8805 (1534.4 examples/sec; 0.083 sec/batch)
        ...


        2017-03-28 19:13:30.626960: precision @ 1 = 0.612

