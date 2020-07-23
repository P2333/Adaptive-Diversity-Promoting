# Adaptive Diversity Promoting Regularization

The adaptive diversity promoting (ADP) method is used to enhance the adversarial robustness of ensemble models. This repository contains the codes for reproducing most of the results proposed in our paper, detailed in:

[Improving Adversarial Robustness via Promoting Ensemble Diversity](https://arxiv.org/pdf/1901.08846.pdf) (ICML 2019)

Tianyu Pang, Kun Xu, Chao Du, Ning Chen and Jun Zhu

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 16.04.3
- GPU: Geforce 1080 Ti or Titan X (Pascal or Maxwell)
- Cuda: 9.0, Cudnn: v7.03
- Python: 2.7.12
- cleverhans: 2.1.0
- Keras: 2.2.4
- tensorflow-gpu: 1.9.0

We also thank the authors of [keras-resnet](https://github.com/raghakot/keras-resnet) for providing their code. Our codes are widely adapted from their repositories.

**Our evaluation is based on cleverhans: 2.1.0. To perform reasonable attacks, please maually modify the command** ```new_image = tf.clip_by_value(input_image + clipped_perturbation, 0, 1)``` **to** ```new_image = tf.clip_by_value(input_image + clipped_perturbation, -0.6, 0.6)``` **in** ```_project_perturbation``` **of the file** ```attacks_tf.py```, **because we substract pixel mean on inputs.**


In the following, we first provide the codes for training our proposed methods and baselines. After that, the evaluation codes, such as attacking, are provided.

## Training codes

### Training baselines and ADP

For training on MNIST dataset, 
```shell
python -u train_mnist.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --dataset='mnist'
```
where the baseline is implemented with alpha_value = beta_value = 0, and the ADP is implemented with the corresponding values in our paper.

For CIFAR10 and CIFAR100, the commands are similar, with following:
```shell
# CIFAR10
python -u train_cifar.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --dataset='cifar10'
# CIFAR100
python -u train_cifar.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --dataset='cifar100'
```

Using the aboved commands, the models used in Table 1 & 2 & 3 can be reproduced.

### Adversarial training with/without ADP
For adversarial training, we use FGSM and PGD methods to craft adversarial examples.
The models can be trained using the following commands:

```shell
# PGD + ADP
python -u advtrain_cifar10.py --attack_method=MadryEtAl --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --dataset='cifar10'
# PGD without ADP
python -u advtrain_cifar10.py --attack_method=MadryEtAl --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --dataset='cifar10' 

# FGSM + ADP
python -u advtrain_cifar10.py --attack_method=FastGradientMethod --lamda=2.0 --log_det_lamda=0.5 --num_models=3 --augmentation=True --dataset='cifar10'
# FGSM without ADP
python -u advtrain_cifar10.py --attack_method=FastGradientMethod --lamda=0.0 --log_det_lamda=0.0 --num_models=3 --augmentation=True --dataset='cifar10'
```

The models used in Table 4 can be reproduced.


## Evaluation codes

The pretrained models are provided for the ensemble of three Resnet-20v1: 

[ADP for 3 models (CIFAR-10)](http://ml.cs.tsinghua.edu.cn/~tianyu/ADP/pretrained_models/ADP_standard_3networks/cifar10_ResNet20v1_model.159.h5)

[ADP with adversarial training for 3 models (CIFAR-10)](http://ml.cs.tsinghua.edu.cn/~tianyu/ADP/pretrained_models/ADP_with_PGDtrain_3networks/cifar10_ResNet20v1_model.124.h5).

### Test in the normal setting
```shell
python -u test_[dataset1]_iterative.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --baseline_epoch=[checkpoint_baseline_epoch] --dataset=[dataset2] --eps=0
```
This command can simultaneously test our method and the baseline method. By substituting the corresponding parameters in the aboved command line, the accuracy can be reproduced in Table 1. The ```checkpoint_epoch``` and ```checkpoint_baseline_epoch``` separately indicate the corresponding checkpoint files which needs to be tested. The ```dataset1``` can be ```mnist, cifar```. The ```dataset2``` can be ```mnist, cifar10, cifar100```. The results can also be obtained from the output logs when training models.

### Test in the adversarial setting
We test our model using different attacking methods, which are implemented by [CleverHans](https://github.com/tensorflow/cleverhans)

For iterative-based attacks: FGSM, BIM, PGD and MIM, the test command is
```shell
python -u test_[dataset1]_iterative.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --baseline_epoch=[checkpoint_baseline_epoch] --attack_method=[attack_method] --dataset=[dataset2] --eps=0.01
```
In this part, ADP and baseline methods are also tested together. For these attack methods, ```--epsilon``` is required to specify the scale for adversarial examples. The ```attack_method``` can be ```FastGradientMethod, BasicIterativeMethod, MadryEtAl, MomentumIterativeMethod```. For code clarity, we only include the codes on CIFAR-10 of other attacks as below.

For optimization-based attacks: C&W, EAD, the test command is
```shell
python -u test_cifar_optimization.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --attack_method=[attack_method] --dataset='cifar10'
```
The ```attack_method``` can be ```CarliniWagnerL2, ElasticNetMethod```.

Note that for JSMA, the attack algorithm provided by cleverhans is not useable. We implement it ourself, which can be used in the following command.
```shell
python -u test_cifar_jsma.py --lamda=[alpha_value] --log_det_lamda=[beta_value] --num_models=3 --augmentation=True --epoch=[checkpoint_epoch] --attack_method='JSMA' --dataset='cifar10'
```
