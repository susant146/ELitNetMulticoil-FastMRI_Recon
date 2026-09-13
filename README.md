# PBC-ELiTNet: Projection-Based Cascaded ELiTNet for MR Image Reconstruction

PyTorch / PyTorch Lightning implementation of **PBC-ELiTNet**, an energy-efficient, lightweight, and computationally thin deep learning model for accelerated multicoil MRI reconstruction from undersampled k-space data, built on the [fastMRI](https://fastmri.med.nyu.edu/) framework.

PBC-ELiTNet extends the **Projection-Based Cascaded U-Net (PBC-UNet)** architecture ([Aghabiglou & Eksioglu, 2021](https://www.sciencedirect.com/science/article/abs/pii/S016926072100225X)) by replacing its U-Net backbone with **ELiTNet**, a layer-wise attention-based lightweight network, while keeping the same projection-based data-consistency (DC) cascade framework. This repository provides the core building blocks of the framework: data loading and transforms, the projection-based DC layer, the gradient/SSIM-based loss functions, and the abstract Lightning model class.

## Abstract

> **Background and Objective:**
> Accelerating magnetic resonance imaging (MRI) through undersampled k-space acquisition can provide a promising approach to reducing scan time and enhancing patient throughput. Deep learning-based reconstruction has shown strong potential in solving this inherently ill-posed inverse problem.
>
> **Methods:**
> We introduce a projection-based cascaded energy efficient, lightweight, and computationally thin network (PBC-ELiTNet) with a data consistency (DC) loss based learning framework, for multicoil MRI reconstruction. The DC layer utilizes a projection-based approach to enforce fidelity to the acquired measurements, thereby improving consistency between the reconstructed images and the original k-space data. ELiTNet, serving as the backbone of the proposed model, is designed to be energy-efficient, lightweight, and computationally thin. It incorporates a novel layer-wise attention mechanism that selectively enhances feature representation by focusing on the most relevant image structures. During training, a consistent gradient-based loss function is used to preserve fine textures and anatomical details.
>
> **Results:**
> Among the variants of ELiTNet, the 16-channel ELiTNet-Medium achieves superior reconstruction quality with approximately half the parameters of the UNet model. Evaluated on the fastMRI dataset across different sequences, PBC-ELiTNet that adopts *ELiTNet-Medium* architecture outperforms several state-of-the-art methods even at a higher acceleration rate R=8 and demonstrates strong generalization to unseen training data. Quantitatively, PBC-ELiTNet achieves a 50.33% parameter reduction and lowers GFLOPs by 36.69% compared to PBC-UNet, resulting in a 1.86x speedup with 46.35% reduced in latency. Furthermore, PBC-ELiTNet requires 2.06x less computation and achieves a 60.04% reduction in latency compared to the transformer-based ReconFormer, highlighting its superior efficiency-performance trade-off.
>
> **Conclusions:**
> The proposed PBC-ELiTNet provides an energy-efficient, lightweight, and computationally thin framework for multicoil MRI reconstruction that achieves high reconstruction quality while substantially reducing model complexity, computational cost, and inference latency.

## Dataset

* [fastMRI](https://fastmri.med.nyu.edu/)

## Repository Structure

```
├── common/                        # Argument parsing, subsampling masks, evaluation metrics, utilities
├── data/                          # fastMRI-style dataset loader and k-space/image transforms
├── dclayer/                       # Projection-based data-consistency (DC) layer used inside the cascade
├── loss_function/                 # Consistent gradient-based loss and combined MSE + gradient + SSIM loss
└── model.py                       # Abstract PyTorch Lightning base class for all reconstruction models
```

## Requirements

* Python 3.8+
* A CUDA-capable GPU is recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

Note: `fastmri` is used for data subsampling/transforms/losses. Install the version compatible with your `torch`/`pytorch-lightning` setup if `pip` does not resolve one automatically.

## Reference

PBC-ELiTNet builds on the projection-based cascaded data-consistency framework introduced in:

> Amir Aghabiglou and Ender M. Eksioglu, "Projection-Based cascaded U-Net model for MR image reconstruction," *Computer Methods and Programs in Biomedicine*, vol. 207, p. 106151, 2021. [https://doi.org/10.1016/j.cmpb.2021.106151](https://www.sciencedirect.com/science/article/pii/S016926072100225X)

## Citation

If you use this code in your project, please cite:

```BibTeX
@article{panigrahi_pbcelitnet,
title = {Attention Guided Projection-based Deep Cascaded Network for Accelerated MRI Reconstruction},
journal = {To be announced},
author = {Susant Kumar Panigrahi and Pradipta Sasmal and Anupam Borthakur and Dipayan Dewan and Debdoot Sheet},
note = {Issue and volume to be added soon}
}
```
