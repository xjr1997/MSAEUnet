# Multi-Scale Attention-Enhanced Deep Learning Model for Ionogram Automatic Scaling
by Li guo, Jiarong Xiong

Code for paper 'Multi-Scale Attention-Enhanced Deep Learning Model for Ionogram Automatic Scaling'

## Dataset
**link to dataset**: [here.](http://www.geophys.ac.cn/ArticleDataInfo.asp?MetaId=205)


## Prerequisites
- [Python3.6](https://www.python.org)
- [Tensorflow2](https://www.tensorflow.org)
- [Segmentation-Models](https://github.com/qubvel/segmentation_models)
- [PyYaml](https://pyyaml.org/)

(If you are not familiar with python, we suggest you to use [Anaconda](https://www.anaconda.com
) to install these prerequisites.)


## Training
You can use `--gpu` argument to specifiy gpu. 
To train a model, first create a configuration file (see example_config.yaml)
Then run
```
python dias_main.py --train --gpu_id 0 --config-file YOUR_CONFIG_PATH
```
Tips: According to feedback that certain implementations of RAdam optimizer have problems in training convergence in this program, switch to Adam optimizer can solve the problem.

## Testing
To test, run
```
python dias_main.py --test --gpu_id 0 --config-file YOUR_CONFIG_PATH
```

### Evaluation
You can evaluate the model's performance by running script:
```
python dias_main.py --eval --config-file YOUR_CONFIG_PATH
```
