# AutoCoG: A Unified Data-Model Co-Search Framework for Graph Neural Networks
Code used for [AutoCoG: A Unified Data-Model Co-search Framework for Graph Neural Networks](https://openreview.net/forum?id=r0zIWWar8gq)


## Installation
We recommend users to use `conda` to install the running environment. Use the following cmd to install the enviroment:
```
conda create --file env.yml
```

## Usage
After installing the enviroment, users can directly train the neural networks by invoking:
```
bash scripts/train_script.sh
```
Immediately they will be prompt the following:
* cuda (please supply the numerical id value of your gpu)
* dataset (refers to ./configurations for supported dataset, supply the dataset name in lowercase here)
* outputdir (supply the output path)
* n_layers (supply the desired number of layers for GNN)
* p_stages (supply the desired number of progressive search stages).

For example:
```
cuda: 0
dataset: texas
outputdir: output/
n_layers: 8
p_stages: 4

```


## Citation
if you find this repo is helpful, please cite
```
@inproceedings{
hoang2022autocog,
title={AutoCoG: A Unified Data-Model Co-Search Framework for Graph Neural Networks},
author={Duc N.M Hoang and Kaixiong Zhou and Tianlong Chen and Xia Hu and Zhangyang Wang},
booktitle={First Conference on Automated Machine Learning (Main Track)},
year={2022},
url={https://openreview.net/forum?id=r0zIWWar8gq}
}
```
