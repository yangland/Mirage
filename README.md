# Mirage

The implementation of paper: **Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning**


Support Apple mps.

# Run

> python main.py --params ./yamls/Mirage/Mirage_nodefense.yaml

# Environment:


> python 3.9.19
>
> pytorch 2.1.1



# Thanks

> 1. BackdoorIndicator: https://github.com/ybdai7/Backdoor-indicator-defense
> 2. A3FL: https://github.com/hfzhang31/A3FL

# Citation
We appreciate it if you would please cite the following paper if you found the repository useful for your work:
```BibTeX
@article{Mirage_2024_Li,
title={Infighting in the Dark: Multi-Labels Backdoor Attack in Federated Learning},
author={Li, Ye and Zhao, Yanchao and Zhu, Chengcheng and Zhang, Jiale},
journal={arXiv preprint arXiv:2409.19601},
year={2024}
}
```

```
Mirage
├─ Mirage
│  ├─ datasets
│  │  └─ MSP_dataloader.py
│  ├─ main.py
│  ├─ models
│  │  ├─ resnet.py
│  │  ├─ simpleNet.py
│  │  └─ vgg.py
│  ├─ participants
│  │  ├─ clients
│  │  │  ├─ BasicClient.py
│  │  │  ├─ BenignClient.py
│  │  │  ├─ MalicilousClient.py
│  │  │  ├─ MirageClient.py
│  │  │  └─ __init__.py
│  │  └─ servers
│  │     ├─ BasicServer.py
│  │     └─ No_defense_Server.py
│  ├─ README.md
│  ├─ saved_models
│  │  └─ ResNet18_cifar10_ptm_2000_benign.pt
│  ├─ utils
│  │  ├─ losses.py
│  │  ├─ utils.py
│  │  └─ visualize.py
│  └─ yamls
│     └─ Mirage
│        └─ Mirage_nodefense.yaml
├─ project_structure.txt
└─ README.md

```