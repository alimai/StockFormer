1）每次回复时都叫我 老板
2）每次回答我问你的问题都要有确定的依据，不确定可以查询上下文
3) 当前工程是一个可以正常工作的python工程，如果发生错误要追溯导致错误的真正原因，通常是一个小的改动导致的

# StockFormer 项目说明

## 项目概述

StockFormer 是一个基于 IJCAI 2023 论文《StockFormer: Learning Hybrid Trading Machines with Predictive Coding》的股票交易强化学习框架。该项目结合了预测编码（Predictive Coding）的前向建模能力和强化学习（RL）在策略灵活性方面的优势，旨在构建高效的混合交易机器。

项目主要分为两个阶段：
1. 表示学习（Representation Learning）：训练关系状态推理模块、长期状态推理模块和短期状态推理模块
2. 策略学习（Policy Learning）：使用 SAC（Soft Actor-Critic）算法训练交易策略

## 项目架构

### 核心组件
- **MySAC**：基于 SAC 算法的强化学习模块，集成了 MAE（Masked Autoencoder）Transformer
- **Transformer**：实现 Transformer 模型，用于预测编码和状态表示学习
- **envs**：股票交易环境，实现了 OpenAI Gym 接口
- **数据处理模块**：包括特征工程、数据标准化等功能

### 主要文件
- `train_rl.py`：主训练脚本，负责整个训练和测试流程
- `MySAC/SAC/MAE_SAC.py`：实现带有掩码自编码器的 SAC 算法
- `envs/env_stocktrading_hybrid_control.py`：股票交易环境定义
- `Transformer/models/transformer.py`：Transformer 模型架构

## 构建和运行

### 环境准备
```bash
git clone https://github.com/gsyyysg/StockFormer.git
cd StockFormer
pip install -r requirements.txt
```

### 数据准备
- 数据存储在 `../data/CSI/` 目录下
- 支持从 Yahoo Finance 下载数据

### 训练流程

#### 第一阶段：表示学习
1. 关系状态推理模块训练：
   ```bash
   cd code/Transformer/script
   sh train_mae.sh
   ```
2. 长期状态推理模块训练：
   ```bash
   sh train_pred_long.sh
   ```
3. 短期状态推理模块训练：
   ```bash
   sh train_pred_short.sh
   ```

#### 第二阶段：策略学习
训练 SAC 模型：
```bash
python train_rl.py
```

### 预训练模型
项目提供了预训练模型，位于 `code/Transformer/pretrained/csi/` 目录下：
- `Short/checkpoint.pth`：短期预测模型
- `Long/checkpoint.pth`：长期预测模型
- `mae/checkpoint.pth`：MAE 模型

## 技术特点

### 掩码自编码器（MAE）
- 在 `_state_transfer()` 方法中实现多种掩码策略
- 支持 'stock'（屏蔽股票）、'feature'（屏蔽特征）、'mixed'（混合）和 'nope'（无掩码）模式
- 掩码比例约为 1%（`max(1, int(stock_num * 0.01))` 或 `max(1, int(feat_dim * 0.01))`）
- 通过掩码-重建机制学习更好的状态表示

### 交易环境
- 支持多股票交易
- 包含协方差矩阵、技术指标等多种状态特征
- 实现了资金管理和风险控制机制

### 模型架构
- 使用 Transformer 作为核心架构
- 结合短期和长期预测模块
- 集成强化学习策略网络

## 配置参数

### 模型参数（MySAC/config.py）
- `MAESAC_PARAMS`：定义 SAC 模型的超参数
- `TECHNICAL_INDICATORS_LIST`：技术指标列表
- `TEMPORAL_FEATURE`：时间序列特征列表

### 训练参数
- `batch_size`：批次大小
- `buffer_size`：经验回放缓冲区大小
- `learning_rate`：学习率
- `d_model`：Transformer 模型维度
- `n_heads`：注意力头数

## 输出结果

训练完成后，结果保存在以下目录：
- `trained_models/`：保存训练好的模型
- `tensorboard_log/`：TensorBoard 日志
- `results/`：训练和测试结果，包括交易动作和资产历史

## 引用信息

如果使用此代码，请引用原论文：
```bibtex
@inproceedings{gaostockformer,
  title={StockFormer: Learning Hybrid Trading Machines with Predictive Coding},
  author={Gao, Siyu and Wang, Yunbo and Yang, Xiaokang},
  booktitle={IJCAI},
  year={2023}
}
```

## 开发约定

- 代码基于 FinRL 框架开发
- 使用标准的 Python 编码规范
- 项目遵循两阶段训练流程：先进行表示学习，再进行策略学习
- 使用 PyTorch 和 Stable Baselines3 实现深度学习和强化学习算法