# Enhancing Knowledge Tracing via learnable filter



## Requirements
```sh
PyTorch==1.7.0
Python==3.8.0
```

### Running
We evaluate our method on four datasets including **ASSISTments2012**, **ASSISTments2009**, **ASSISTments2015** and **ASSISTments2017**.


#### ASSISTments2009
```
python main.py --dataset 'assist2009_pid'
```

#### ASSISTments2015
```
python main.py --dataset 'assist2015'
```

#### ASSISTments2017
```
python main.py --dataset 'assist2017_pid'
```
Evaluated results (AUC scores) will be saved in **statics_test_result.txt**, **assist2009_pid_test_result.txt**, **assist2015_test_result.txt**, and **assist2017_pid_test_result.txt**, respectively.


## Acknowledgments
Code and datasets are borrowed from [AKT](https://github.com/arghosh/AKT). Early stopping implementation is modified from [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch).


