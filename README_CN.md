# GSDFuse

[ [En](README.md) | [中](#chinese) ]

<a id="chinese"></a>

基于图神经网络的社交媒体隐写术检测框架，包含GIN、GAU、三元组损失和SMOTE采样等核心组件。

## 项目概述

GSDFuse是一个用于社交媒体隐写术检测的框架，利用图神经网络对社交网络中的用户行为进行分析，以识别可能包含隐藏信息的内容。该项目采用了多种先进技术，包括：

- GIN (Graph Isomorphism Network)：增强图结构特征提取
- GAU (Graph Attention Unit)：注意力机制增强节点特征提取
- 三元组损失 (Triplet Loss)：提高分类边界清晰度
- SMOTE采样：解决类别不平衡问题

## 项目结构

```
GSDFuse/
├── method/                 # 核心模型和算法实现
│   ├── models.py           # 模型定义
│   ├── minibatch.py        # 批次处理
│   ├── smote_sampler.py    # SMOTE采样实现
│   └── utils.py            # 工具函数
├── configs/                # 配置文件目录
│   └── gat_192_8_with_graph_improved.yml  # 示例配置
├── main.py                 # 主程序入口
├── globals.py              # 全局变量和参数定义
├── utils.py                # 通用工具函数
└── metric.py               # 评估指标计算
```

## 安装与环境

### 依赖项

本项目需要以下依赖项（推荐版本用于复现结果）：

```
python == 3.8.11
pytorch == 1.8.0
cython == 0.29.24
g++ == 5.4.0  # 用于Cython扩展的C++编译器
numpy == 1.20.3
scipy == 1.6.2
scikit-learn == 0.24.2
imbalanced-learn  # SMOTE实现
pyyaml == 5.4.1
```

完整依赖可通过以下命令安装：

```bash
pip install -r requirements.txt
```

### 运行前准备

本项目包含Cython模块，需要在开始训练前进行编译。请在项目根目录运行以下命令编译模块：

```bash
python setup.py build_ext --inplace
```

## 使用方法

### 实验指南

在以下命令中，请将 `${relative_data_dir}` 替换为您的实际数据集路径。例如：
```
relative_data_dir=./data/reddit-ac-1-onlyends_with_isolated_bi_10percent_hop1
```

#### 基准模型

**仅文本基准（无图结构）**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/no_graph.yml --no_graph --repeat_time 1
```

**TS-CSW+GRAPH**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0
```

**TGCA**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_TGCA
```

**CATS**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_CATS --use_GAaN
```

#### 我们的组件

**GAU（图注意力单元）**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_gau
```

**GIN（图同构网络）**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_GIN
```

**三元组损失**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_triplet_loss --use_hard_mining
```

注意：上述命令默认使用 **semi-hard** 挖掘策略。也可以通过 `--mining_strategy` 参数指定其他策略：

```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_triplet_loss --use_hard_mining --mining_strategy hard
```

可用的挖掘策略包括：`hard`、`semi-hard`、`random`。

**SMOTE**：
```bash
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_no_smote.yml --repeat_time 1 --gpu 0 --use_smote
```

有关方法和结果的详细信息，请参阅我们的论文。

## 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。
