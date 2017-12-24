# Tensorflow Queue Example (no longer maintainted)
*This repo is no longer maintained. If you are interested in playing with high-level tensorflow APIs, please check out [this Gist](https://gist.github.com/markdtw/ef1ec76e6be9104316c454455754af04), it's pretty much a cleaner version of tensorflow official tutorial with queue loader as well, should be easy to follow.*

An organized and simple example for loading data in queues using Tensorflow, with CIFAR-10 as simulated input data, compatible with tensorflow >= r1.0.

Provide `cnn.py` and `alexnet.py` as naive CNN architecture for training/testing with queues.

The standard way (with TFRecords) to run queue in Tensorflow:
- Generate a single file containing both training images and labels in `.tfrecords` format.
- Read from `.tfrecords` file:
  - use `tf.string_input_producer` to create filename queue
  - use a custom reader to parse single example from `.tfrecords`
  - generate a queue runner with `tf.train.shuffle_batch`

Good luck!
## Prerequisites
- Python 2.7+
- [NumPy](http://www.numpy.org/)
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)

## Data
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Preparation
1. Clone this repo, create `data_log/` folder:
```bash
git clone https://github.com/markdtw/tensorflow-queue-example.git
cd tensorflow-queue-example
mkdir data_log
```
2. Download and extract [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).  
   Save it in `data_log` folder.

3. Go through `queue_loader.py` from bottom to top for more information.


## Train
Train a very simple CNN from scratch with default settings:
```bash
python main.py --train
```
First time training will dump a `train_package.tfrecords` file in `data_log/` folder.

Check out tunable arguments:
```bash
python main.py
```

## Test
```bash
python main.py --test
```
First time testing will dump a `test_package.tfrecords` file in `data_log/` folder.

## Result
The result will be something like this:
```bash
...
...
epoch:  3, step: 90, loss: 2.8682
epoch:  3, step: 97, loss: 2.8901, epoch  3 done.
W tensorflow/core/framework/op_kernel.cc:1152] Out of range: RandomShuffleQueue '_2_shuffle_batch/random_shuffle_queue' is closed and has insufficient elements (requested XXX, current size XXX)
        [[Node: shuffle_batch = QueueDequeueManyV2[component_types=[DT_FLOAT, DT_INT32], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/cpu:0"](shuffle_batch/random_shuffle_queue, shuffle_batch/n)]]
...
...
        [[Node: shuffle_batch = QueueDequeueManyV2[component_types=[DT_FLOAT, DT_INT32], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/cpu:0"](shuffle_batch/random_shuffle_queue, shuffle_batch/n)]]

Done training, epoch limit: 3 reached.
```
These warning messages cannot be caught which is similar to [this issue](https://github.com/tensorflow/tensorflow/issues/8330) but it's totally fine, the error is caught and handled later on.

## Resources
- [Tensorflow Official Document for Reading Data](https://www.tensorflow.org/programmers_guide/reading_data)
- [Tensorflow Official Document for Threading and Queues](https://www.tensorflow.org/programmers_guide/threading_and_queues)
- [Tensorflow Official Code for Generating a TFRecords File](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
- [Tensorflow Official Code for Loading a TFRecords File](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py)

