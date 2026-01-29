# 📄 关于 `off_policy_algorithm.py` 多副本问题的分析报告

## 1. 背景概述
在当前工程中存在两个同名的 `off_policy_algorithm.py` 文件，分别位于自定义模块目录和本地标准库目录。经过代码审计和调用链分析，确认这两个文件均处于激活状态，但服务于不同的算法逻辑。

### 文件路径
1.  **自定义版本**：`code/MySAC/SAC/off_policy_algorithm.py`
2.  **标准版本**：`code/stable_baselines3/common/off_policy_algorithm.py`

---

## 2. 调用关系分析

| 文件位置 | 主要调用者 | 适用场景 |
| :--- | :--- | :--- |
| `MySAC/SAC/` | `code/MySAC/SAC/MAE_SAC.py` | 运行 `StockFormer` 核心模型 `MAE_SAC` 时使用。 |
| `stable_baselines3/` | `stable_baselines3` 下的 `dqn`, `td3`, `sac` 等标准算法 | 运行 SB3 原生算法或进行基准测试时使用。 |

在执行 `train_rl.py` 训练 `maesac` 模型时，系统会优先通过显式导入使用 `MySAC` 目录下的自定义版本。

---

## 3. 核心代码差异

这两个文件的代码重合度超过 99%，唯一的**关键差异**位于策略网络（Policy）的初始化逻辑中：

### 差异代码对比 (约第 222 行)

*   **自定义版本 (MySAC)**：
    ```python
    self.policy = self.policy_class(
        self.hidden_state_space,  # 使用 Transformer 转换后的隐藏状态空间
        self.action_space,
        self.lr_schedule,
        **self.policy_kwargs,
    )
    ```
*   **标准版本 (SB3)**：
    ```python
    self.policy = self.policy_class(
        self.observation_space,  # 使用环境原始的观察空间
        self.action_space,
        self.lr_schedule,
        **self.policy_kwargs,
    )
    ```

---

## 4. 为什么要进行这种定制？

这是 `StockFormer` 架构的核心设计决定的：

1.  **特征降维**：原始股票数据（包含协方差矩阵和技术指标）维度极高。本项目使用 Transformer (MAE) 作为特征提取器，将高维输入映射为低维的 `hidden_state`。
2.  **维度匹配**：策略网络（Actor/Critic）的输入层必须匹配 `hidden_state` 的维度（通常为 128），而不是原始 `observation_space` 的维度。
3.  **架构隔离**：为了不破坏原生 `stable_baselines3` 库对普通任务的支持，开发者选择在 `MySAC` 目录下复制并修改了基类，实现了针对 `StockFormer` 任务的专用逻辑。

---

## 5. 结论与风险评估

### 结论
**MySAC 目录下的版本没有问题，且是项目运行所必须的。** 它确保了强化学习的决策层能正确接收来自预训练 Transformer 的特征向量。

### 潜在风险
*   **维护冗余**：两个文件高度重复。如果未来需要修改 Off-Policy 算法的通用逻辑（如 `collect_rollouts` 采样逻辑），必须在两个文件中同步修改。
*   **同步建议**：若无特殊需求，建议保持这两个文件除了 `hidden_state_space` 初始化逻辑外的一致性。

---
*文档更新日期：2026年1月29日*
