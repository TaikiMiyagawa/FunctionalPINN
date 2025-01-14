# 😎 Functional PINN [NeurIPS 2024] 😎 <!-- omit in toc -->
<!-- 
Table of contents: https://qiita.com/eyuta/items/b1a53f3da8c5f8e7f41d  
NeurIPS code guidelines: https://github.com/paperswithcode/releasing-research-code   
-->

### 🌟Work in progress! Please use ./full_code for now.🌟 <!-- omit in toc -->

This is the official implementation of physics-informed neural networks (PINNs) for functional differential equations (Functional PINN) proposed in


### ["Physics-informed Neural Networks for Functional Differential Equations: Cylindrical Approximation and Its Convergence Guarantees" (NeurIPS 2024).](./full_paper.pdf) <!-- omit in toc -->

- Full paper: [`./fullpaper.pdf`](./full_paper.pdf)
- Full paper at OpenReview: https://openreview.net/forum?id=H5z0XqEX57
- arXiv preprint: https://arxiv.org/abs/2410.18153
- bibtex: please see Citation at the bottom of this page.

# Table of Contents <!-- omit in toc -->

- [1. Introduction: Overall architecture](#1-introduction-overall-architecture)
- [2. Requirements](#2-requirements)
- [3. Training](#3-training)
- [4. Files and Directories](#4-files-and-directories)
- [5. Citation](#5-citation)
- [Todo](#todo)

# 1. Introduction: Overall architecture

The overall architecture is shown in Figure 1.
We first approximate functional differential equations (FDEs) using the cylindrical approximation, leading to high-dimensional PDEs (we implement these PDEs), which are then solved with PINNs (Figure 3).
![Overall architecture](./imgs/figure1.png)
![PINN](./imgs/figure3.png)

# 2. Requirements

Please see [`requirements.txt`](./requirements.txt) and run `$ pip install -r requirements.txt` to install exactly the same libraries used in our environment. All the libraries, however, are not necessary if you just want to run `./train.py`. Specifically, we used:

- Python 3.11
- PyTorch 2.2.0
- Numpy 1.26.0

# 3. Training

`$ python train.py`

# 4. Files and Directories

- [`./configs`](./configs)
- [`./dataprocesses`](./dataprocesses)
- [`./losses`](./losses)
- [`./models`](./models)
- [`./optimizers`](./optimizers)
- [`./utils`](./utils)
- [`./train.py`](./train.py)
- [`./training_controller.py`](./training_controller.py)
- [`./full_code`](./full_code)
  - All codes submitted to NeurIPS 2024. Just for reference. If you miss something in the top directory, you could find it here, or contact us.
- [`full_paper.pdf`](./full_paper.pdf)
  - The full paper of our work. This is exactly the same as the one submitted to OpenReview but is different from the arXiv paper.

# 5. Citation

```
@inproceedings{
miyagawa2024physicsinformed,
title={Physics-informed Neural Networks for Functional Differential Equations: Cylindrical Approximation and Its Convergence Guarantees},
author={Taiki Miyagawa and Takeru Yokota},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=H5z0XqEX57}
}
```
