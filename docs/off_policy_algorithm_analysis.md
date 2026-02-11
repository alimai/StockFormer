# 📄 深度强化学习算法架构适配分析

## 1. 核心挑战：维度不匹配
在 StockFormer 架构中，Transformer (MAE) 提取的隐藏特征维度（如 128 或 256）与环境原始观察空间维度（包含协方差矩阵和技术指标）不匹配。

## 2. 解决方案：动态观察空间 Hook 机制
本工程在 `MySAC/SAC/MAE_SAC.py` 中通过重写 `_setup_model` 方法，实现了一种对原生 `stable_baselines3` 库零侵入的 Hook 机制。

### 实现逻辑
```python
def _setup_model(self) -> None:
    # 1. 暂存环境原始观察空间 (Gymnasium 格式)
    original_obs_space = self.observation_space
    
    # ... 初始化 ReplayBuffer (存储原始高维数据) ...

    # 2. 核心 Hook：临时切换观察空间维度
    # 切换为 Transformer 输出的隐藏状态维度，确保 Actor/Critic 网络输入层正确初始化
    self.observation_space = self.hidden_state_space
    
    # 3. 初始化 Policy
    self.policy = self.policy_class(...)
    
    # 4. 立即还原原始观察空间
    # 确保环境交互采样流程不受影响
    self.observation_space = original_obs_space
```

## 3. Gymnasium 适配
随着库升级到 `Gymnasium`，该 Hook 机制依然稳健。`ReplayBuffer` 会自动识别 `gymnasium.spaces.Box` 空间。模型在 `predict` 阶段通过 `_state_transfer` 自动完成从原始空间到隐藏空间的实时转换。

## 4. 架构优势
- **原生兼容**：无需修改 SB3 源码，支持 `pip install` 标准安装。
- **内存高效**：`ReplayBuffer` 存储原始数据，仅在 `train()` 过程中由 `state_transformer` 实时转换为隐藏表示，避免了在 buffer 中存储大量高维向量。
- **逻辑收敛**：所有的维度变换和特征提取逻辑都封装在 `MAE_SAC.py` 内部。

---
*文档更新日期：2026年2月11日*
