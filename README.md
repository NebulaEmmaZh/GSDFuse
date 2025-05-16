# GSDFuse

[ [En](#english) | [中](README_CN.md) ]

---

<a id="english"></a>

A Graph Neural Network Framework for Social Media Steganography Detection, including GIN, GAU, Triplet Loss, and SMOTE components.

## Project Overview

GSDFuse is a framework for detecting steganography in social media, utilizing graph neural networks to analyze user behavior in social networks to identify content that may contain hidden information. The project employs various advanced techniques, including:

- GIN (Graph Isomorphism Network): Enhanced graph structure feature extraction
- GAU (Graph Attention Unit): Attention mechanism enhanced node feature extraction
- Triplet Loss: Improved classification boundary clarity
- SMOTE sampling: Solving class imbalance issues

## Project Structure

```
GSDFuse/
├── method/                 # Core model and algorithm implementation
│   ├── models.py           # Model definitions
│   ├── minibatch.py        # Batch processing
│   ├── smote_sampler.py    # SMOTE sampling implementation
│   └── utils.py            # Utility functions
├── configs/                # Configuration file directory
│   └── gat_192_8_with_graph_improved.yml  # Example configuration
├── main.py                 # Main program entry
├── globals.py              # Global variables and parameter definitions
├── utils.py                # General utility functions
└── metric.py               # Evaluation metric calculations
```

## Installation & Environment

### Dependencies

The project requires the following dependencies (recommended versions for reproducibility):

```
python == 3.8.11
pytorch == 1.8.0
cython == 0.29.24
g++ == 5.4.0  # C++ compiler for Cython extensions
numpy == 1.20.3
scipy == 1.6.2
scikit-learn == 0.24.2
imbalanced-learn  # SMOTE implementation
pyyaml == 5.4.1
```

Complete dependencies can be installed with:

```bash
pip install -r requirements.txt
```

### Before Running

This project includes a Cython module that needs to be compiled before training can start. Compile the module by running the following command from the project's root directory:

```bash
python setup.py build_ext --inplace
```

## Usage

### Experiment Guide

In the following commands, replace `${relative_data_dir}` with your actual dataset path. For example:
```
relative_data_dir=./data/reddit-ac-1-onlyends_with_isolated_bi_10percent_hop1
```

#### Baselines

**Text-only baseline (no graph structure)**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/no_graph.yml --no_graph --repeat_time 1
```


**TS-CSW+GRAPH**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0
```

**TGCA**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_TGCA
```

**CATS**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_CATS --use_GAaN
```

#### Our Components

**GAU (Graph Attention Unit)**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_gau
```

**GIN (Graph Isomorphism Network)**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_GIN
```

**Triplet Loss**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_triplet_loss --use_hard_mining
```

Note: The above command uses the **semi-hard** mining strategy by default. You can specify other strategies using the `--mining_strategy` parameter:

```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_triplet_loss --use_hard_mining --mining_strategy hard
```

Available mining strategies: `hard`, `semi-hard`, `random`.

**SMOTE**:
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_smote
```


For details of the methods and results, please refer to our paper.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

