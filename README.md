# GSDFuse

### Capturing Cognitive Inconsistencies from Multi-Dimensional Weak Signals in Social Media Steganalysis

[![ICASSP 2026](https://img.shields.io/badge/ICASSP-2026-blue)](https://ieeexplore.ieee.org/document/11465080)
[![arXiv](https://img.shields.io/badge/arXiv-2505.17085-b31b1b)](https://arxiv.org/abs/2505.17085)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**English** | [简体中文](README_CN.md)

Code for **GSDFuse**, a framework for detecting steganographic messages in social media conversations through textual and graph structural signals.

[Overview](#overview) · [Installation](#installation) · [Data preparation](#data-preparation) · [Training](#training) · [Experiments](#experiments) · [Citation](#citation)

## Overview

GSDFuse treats social media steganalysis as **node classification**: messages are nodes, reply relationships form edges, and the model predicts whether each message contains hidden information. It combines message semantics with conversational structure to identify weak, context-dependent signs of steganography.

[![GSDFuse architecture: hierarchical textual and graph representations, data augmentation, adaptive feature fusion, and discriminative learning.](assets/gsdfuse-framework.png)](assets/gsdfuse-framework.png)

*GSDFuse framework. Extracted from Figure 2 of the [arXiv preprint](https://arxiv.org/pdf/2505.17085). Click the figure to view it at full resolution.*

| Component | Purpose | Implementation switch |
| --- | --- | --- |
| Text and local graph encoding | Represent message content and neighboring interactions | Graph encoding is enabled by default |
| Graph Isomorphism Network (GIN) | Add a structural summary of the sampled graph | `--use_GIN` |
| Gated Attention Unit (GAU) | Adaptively fuse textual and structural features | `--use_gau` |
| SMOTE | Augment minority-class training samples | `--use_smote` |
| Triplet loss | Encourage separation between cover and stego representations | `--use_triplet_loss` |

The [preprint](https://arxiv.org/abs/2505.17085) reports experiments on Sina Weibo, Twitter, and Reddit with HC, AC, and ADG steganography. Refer to the paper for results and experimental settings.

## Installation

The commands below use Bash and target Linux with a C++ compiler; `setup.py` explicitly selects `g++`.

```bash
git clone https://github.com/NebulaEmmaZh/GSDFuse.git
cd GSDFuse

conda create -n gsdfuse python=3.8.20 -y
conda activate gsdfuse

python -m pip install -r requirements.txt
python -m pip install flash-pytorch
python -m pip check

python setup.py build_ext --inplace
```

The versions declared in [requirements.txt](requirements.txt) are:

| Package | Version |
| --- | --- |
| PyTorch | 2.4.1 |
| PyTorch Geometric | 2.3.1 |
| NumPy | 1.20.3 |
| SciPy | 1.6.2 |
| scikit-learn | 1.3.2 |
| imbalanced-learn | 0.12.4 |
| PyYAML | 5.4.1 |
| Cython | 0.29.24 |

`flash-pytorch` provides the `flash_pytorch.GAU` imported by the model, including when the GAU switch is disabled. This dependency is currently missing from `requirements.txt`, and its original experiment version is not recorded. The instructions reflect the declared dependencies; the complete environment has not been newly validated. For GPU training, use a PyTorch build compatible with your CUDA runtime.

The build creates the `cython_sampler`, `cython_utils`, and `norm_aggr` extensions. Compile them before running the training code.

## Data preparation

Training expects a **preprocessed graph dataset**. The current repository does not include datasets, pretrained checkpoints, or a script that converts raw conversations into the required format.

```text
data/<dataset>/
├── adj_full.npz
├── adj_train.npz
├── feats.npy
├── class_map.json
└── role.json
```

| File | Expected contents |
| --- | --- |
| `adj_full.npz` | SciPy sparse CSR adjacency matrix of shape `N × N` for the full graph |
| `adj_train.npz` | CSR matrix of the same shape, containing edges between training nodes for graph sampling |
| `feats.npy` | Node-aligned NumPy array; with the default `--sentence_embed cnn`, each row is a fixed-length sequence of integer token IDs |
| `class_map.json` | Mapping from stringified node IDs to labels; for binary steganalysis, use `0` for cover and `1` for stego |
| `role.json` | Node split indices under `tr`, `va`, and `te` |

Node IDs must be contiguous integers from `0` to `N - 1`, consistently aligned across all files. An illustrative split is:

```json
{"tr": [0, 1, 2], "va": [3], "te": [4, 5]}
```

This only demonstrates the split format, not a runnable sample dataset. See [`load_data` and `parse_n_prepare` in utils.py](utils.py) for loading and token-remapping behavior.

Use a path **relative to the repository root** for `--data_prefix`: the current loader prepends `./`.

## Training

Run from the repository root and replace the dataset placeholder:

```bash
DATA_DIR="data/<dataset>"
```

### Combined component example

This enables GIN, GAU, SMOTE, and triplet loss with semi-hard mining:

```bash
python main.py \
  --data_prefix "$DATA_DIR" \
  --train_config train_config/gat_192_8_no_smote.yml \
  --gpu 0 \
  --repeat_time 1 \
  --use_GIN \
  --use_gau \
  --use_smote \
  --use_triplet_loss \
  --use_hard_mining \
  --mining_strategy semi-hard
```

This is a component-combination example. To reproduce a paper table, match its dataset construction, embedding rate, steganographic ratio, configuration, and repetition count.

### Configuration and outputs

The example YAML uses a hidden dimension of 192, 8 attention heads, learning rate 0.01, dropout 0.2, and random-walk subgraph sampling, with a maximum of 200 epochs. Training selects checkpoints by validation F1 and stops after 20 validation checks without improvement.

| Option | Meaning | Default |
| --- | --- | --- |
| `--data_prefix` | Prepared dataset directory | Required |
| `--train_config` | Training YAML file | Required |
| `--gpu` | GPU index | `0` |
| `--repeat_time` | Number of repeated runs | `10` |
| `--sentence_embed` | Text encoder | `cnn` |
| `--dir_log` | Output base directory | `./test` |
| `--mining_strategy` | Triplet strategy when `--use_hard_mining` is enabled | `semi-hard` |

Mining strategies are `random`, `semi-hard`, and `hard`. See [globals.py](globals.py) for all arguments, including SMOTE options.

Default outputs:

- **Metrics:** `test/log_train/<dataset>/<timestamp>-graphsaint-<commit>/result.json`, containing means and standard deviations across repeated runs.
- **Checkpoint:** `test/pytorch_models/saved_model_<timestamp>.pkl`. Repeated runs within one invocation share this path; separate checkpoints are not retained for each repetition.

Reported metrics include accuracy, precision, recall, F1, and micro/macro F1. Use `--repeat_time 1` for an initial check and the paper's protocol for comparisons.

## Experiments

### Baselines

These commands select the baseline branches implemented in this repository:

```bash
# Text-only (TS-CSW).
python main.py --data_prefix "$DATA_DIR" --train_config train_config/no_graph.yml --no_graph --gpu 0 --repeat_time 1

# Text + graph.
python main.py --data_prefix "$DATA_DIR" --train_config train_config/gat_192_8_no_smote.yml --gpu 0 --repeat_time 1

# TGCA.
python main.py --data_prefix "$DATA_DIR" --train_config train_config/gat_192_8_no_smote.yml --gpu 0 --repeat_time 1 --use_TGCA

# CATS.
python main.py --data_prefix "$DATA_DIR" --train_config train_config/gat_192_8_no_smote.yml --gpu 0 --repeat_time 1 --use_CATS --use_GAaN
```

### Component studies

For additive studies, append the desired switches to the text + graph baseline:

| Addition | Append to the baseline command |
| --- | --- |
| GIN | `--use_GIN` |
| GAU | `--use_gau` |
| SMOTE | `--use_smote` |
| Triplet loss with random sampling | `--use_triplet_loss` |
| Triplet loss with semi-hard mining | `--use_triplet_loss --use_hard_mining --mining_strategy semi-hard` |
| Triplet loss with hard mining | `--use_triplet_loss --use_hard_mining --mining_strategy hard` |

For removal ablations, start with the combined example and remove the corresponding switches. Use `gat_192_8_no_smote.yml` when controlling SMOTE through the command line: `gat_192_8_with_graph_improved.yml` enables SMOTE in YAML even without `--use_smote`.

## Repository structure

```text
GSDFuse/
├── assets/
│   └── gsdfuse-framework.png  # Framework figure extracted from the paper
├── method/
│   ├── models.py          # Models, feature fusion, and losses
│   ├── layers.py          # Graph and neural network layers
│   ├── minibatch.py       # Subgraph mini-batches
│   └── smote_sampler.py   # Synthetic minority sample generation
├── train_config/          # Model and sampling configurations
├── main.py                # Training and evaluation entry point
├── globals.py             # Arguments and random seeds
├── utils.py               # Data loading and configuration parsing
├── metric.py              # Evaluation metrics
├── graph_samplers.py      # Graph sampling interfaces
├── cython_sampler.pyx     # Compiled graph samplers
├── cython_utils.pyx       # Compiled sampling utilities
├── norm_aggr.pyx          # Compiled aggregation normalization
├── setup.py               # Cython extension build
└── requirements.txt       # Declared Python dependencies
```

## Citation

If you use this code or method in your research, please cite the ICASSP 2026 paper:

```bibtex
@inproceedings{huang2026gsdfuse,
  title     = {{GSDFuse}: Capturing Cognitive Inconsistencies from Multi-Dimensional Weak Signals in Social Media Steganalysis},
  author    = {Huang, Kaibo and Yang, Zhongliang and Wei, Yukun and Zhang, Zipei and Zhou, Linna},
  booktitle = {ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  doi       = {10.1109/ICASSP55912.2026.11465080},
  url       = {https://ieeexplore.ieee.org/document/11465080}
}
```

## License and questions

The code is released under the [MIT License](LICENSE). For implementation or reproduction questions, please [open an issue](https://github.com/NebulaEmmaZh/GSDFuse/issues).
