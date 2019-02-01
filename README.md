# Tensorflow Boilerplate using Estimator API

TensorFlow boilerplate sample code using ```tf.estimator```, ```tf.data```, ```tf.train``` hooks
to build flexible and efficient input pipelines with
simplified training in a **single-node multi-gpu** setting.


## Development motive

Many people know the TensorFlow v2.0 will be coming! So I can't guarantee this code would be also helpful in v2.0.

Anyway, when we look back on machine learning & AI sessions in Google I/O 2018 (last year),
the google emphasize their distributed training optimization & input pipe-lining with TensorFlow.

This highlighted features can be easily used by using Estimator API,
but I saw many people don't use this, but just use low-level modules.
So I made this template for better training experiments & relatively simple code than
other multi-gpu training codes.

Currently, there are some nice packages as [NCCL(for multi-gpu communication)](https://github.com/NVIDIA/nccl),
[Horovod(easy distributed training for including other python DL frameworks)](https://github.com/uber/horovod),
this code snippet could run **without any external dependencies**.

Plus, nice codes are maintained by [tensorflow/models repository](https://github.com/tensorflow/models).
I referenced the official code a lot.
However, I think that code structure is not appropriate for multi-model multi-dataset training.
It is for single-model multi-dataset in my opinion.
When I do some AI experiments, usually I try various models for various datasets to find which model fits well overall. 

## Prerequisites

The current version is tested with
 - python 3.6
 - TensorFlow v1.12 (install cpu/gpu version in your favor)
 
For more details, see ```requirements.py``` file


## Explanations

### Directory Structure

```
.
├── README.md
├── checkpoints
│   ├── 2019-02-01-21_26_06
│   │   ├── checkpoint
│   │   ├── eval
│   │   ├── events.out.tfevents.1549023969.local
│   │   ├── graph.pbtxt
│   │   ├── model.ckpt-0.data-00000-of-00001
│   │   ├── model.ckpt-0.index
│   │   ├── model.ckpt-0.meta
│   │   ├── model.ckpt-4000.data-00000-of-00001
│   │   ├── model.ckpt-4000.index
│   │   ├── model.ckpt-4000.meta
├── data
│   └── cifar10
│       ├── cifar-10-batches-py
│       ├── cifar-10-python.tar.gz
│       ├── eval.tfrecords
│       ├── train.tfrecords
│       └── validation.tfrecords
├── datasets
│   ├── __init__.py
│   ├── cifar10.py
│   └── dataset_interface.py
├── flags
│   ├── _base.py
│   ├── _benchmark.py
│   ├── _conventions.py
│   ├── _performance.py
│   └── core.py
├── generate_cifar10_tfrecords.py
├── hooks
│   ├── __init__.py
│   ├── hooks_helper.py
│   ├── metric_hook.py
│   └── throughput_hook.py
├── logger
│   ├── __init__.py
│   └── logger.py
├── models
│   ├── __init__.py
│   └── simple_net.py
├── requirements.txt
├── supervisor.py
├── utils
│   └── distribution_utils.py
```

#### supervisor.py
Many people may know supervisor in TensorFlow.
This API was deprecated, but many people loved this for convenience so I named like that.
It is a kind of main file, where you select model, select dataset, and train & evaluate model.

#### data
where you put original data (csv, image, label, etc)

#### datasets
where you parse data files. Like ```dataset_interface.py``` file, you should implement data parsing function,
get training set, get validation set input function for each different dataset.

#### flags
many developers define flags with ```tf.app.flags``` or ```argparse.ArgumentParser()```
or other config(.cfg, .json) file.
But when flags are too many, it is hard to maintain.
So, the flags are defined & managed in different files. 

#### hooks
you can add some hooks to print / log / save results during training.

#### logger
logging TensorFlow Version, CPU, GPU, memory information, other configuration params

#### models
where you define model. You should define model & model_fn function to use in supervisor

#### utils
there is a single file named distribution_utils. This file helps you for setting multi-gpu training.

## Setup (for cifar-10 training / evaluation)

1. download cifar-10 data from [official page](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and unzip
2. ```python generate_cifar10_tfrecords.py --data_dir=[downloaded_and_unzipped_directory]```
3. define appropriate flags in ```supervisor.py``` OR you could give some params in command line with ```--```
3. ```python supervisor.py train_and_eval : for training and validation```


## Author
 - **dev-strender** / [@dev-strender](https://github.com/dev-strender)
 
## References
 - [tensorflow/models repository](https://github.com/tensorflow/models)