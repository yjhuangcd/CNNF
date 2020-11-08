# CNNF
<img align="center" src="CNNF.png" width="750">

## Introduction

Neural networks are vulnerable to input perturbations such as additive noise and adversarial attacks. In contrast, human perception is much more robust to such perturbations. The Bayesian brain hypothesis states that human brains use an internal generative model to update the posterior beliefs of the sensory input. This mechanism can be interpreted as a form of self-consistency between the maximum a posteriori (MAP) estimation of an internal generative model and the external environment. Inspired by such hypothesis, we enforce self-consistency in neural networks by incorporating generative recurrent feedback. We instantiate this design on convolutional neural networks (CNNs). The proposed framework, termed Convolutional Neural Networks with Feedback (CNN-F), introduces a generative feedback with latent variables to existing CNN architectures, where consistent predictions are made through alternating MAP inference under a Bayesian framework. In the experiments, CNN-F shows considerably improved adversarial robustness over conventional feedforward CNNs on standard benchmarks.

For more details please see our [NeurIPS 2020 paper](https://arxiv.org/abs/2007.09200).

## Contents

This directory includes the Pytorch implementation of CNN-F (CNN with feedback).
`layers.py` contains the feedback layers that are used to build CNN models.
`models_mnist.py` and `models_cifar.py` contain the CNN architecture that we train the CNN-F models on. 
`train.py` contains the codes of adversarial training on CNN-F. 
`test.py` contains the codes to evaluate adversarial robustness of CNN-F.
The pretrained models are in `models`.

## Requirements
The codes are tested under NVIDIA container image for PyTorch, release 20.03.

*   numpy==1.18.1
*   torch==1.5.0
*   torchvision==0.6.0
*   advertorch==0.2.3


## Usage

To train a CNN-F model on CIFAR-10 under adversarial training, run: `sh ./run_train_cifar.sh`.

To test the adversarial robustness of a pretrained model, run: `python test.py`.


## Citation

If you find this useful for your work, please consider citing

```
@article{huang2020cnnf,
  title={Neural Networks with Recurrent Generative Feedback},
  author={Huang, Yujia and Gornet, James and Dai, Sihui and Yu, Zhiding and Nguyen, Tan and Tsao, Doris Y and Anandkumar, Anima},
  journal={NeurIPS},
  year={2020}

}
```