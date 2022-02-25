# CRAB: Certified Patch Robustness Against Poisoning-based Backdoor Attacks

This repository contains the code for replicating the results of our paper:

**CRAB: Certified Patch Robustness Against Poisoning-based Backdoor Attacks** </br>
*Huxiao Ji, Jie Li, Chentao Wu*

Check script folders for \*.sh files for each corresponding experiments.


**CIFAR-10 in fixed setting**

```console
cd .\scripts\cifar10\fixed\
bash train.sh <gpu id(s)> <base/small/tiny> <ablation size>
bash certify.sh <gpu id(s)> <base/small/tiny> <ablation size> <batch size>
```
**CIFAR-10 in randomized setting**

```console
cd .\scripts\cifar10\randomized\
bash train.sh <gpu id(s)> <base/small/tiny> <ablation size> <patch size>
bash certify.sh <gpu id(s)> <base/small/tiny> <ablation size> <patch size> <batch size>
```
**MNIST in fixed setting**

```console
cd .\scripts\mnist\fixed\
bash train.sh <gpu id(s)> <ablation size>
bash certify.sh <gpu id(s)> <ablation size>
```
**MNIST in randomized setting**

```console
cd .\scripts\mnist\randomized\
bash train.sh <gpu id(s)> <ablation size> <patch size>
bash certify.sh <gpu id(s)> <ablation size> <patch size>
```
