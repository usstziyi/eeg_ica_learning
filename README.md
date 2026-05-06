# EEG 预处理中的 ICA —— 从零到实战

## 学习路线图

```
Unit 0  →  环境准备（MNE 安装 + sample data 下载）
Unit 1  →  ICA 数学直觉（鸡尾酒会问题、盲源分离）
Unit 2  →  ICA 算法原理（FastICA、非高斯性最大化）
Unit 3  →  EEG 伪迹识别（眨眼、眼动、心跳、肌电）
Unit 4  →  MNE ICA 实战（拟合 / 识别 / 剔除流水线）
Unit 5  →  完整演练（mne-sample-data 端到端）
```

## 使用方法

```bash
cd notebooks/
jupyter notebook    # 或 jupyter lab
```

按 Unit 0 → 5 顺序运行，每个 notebook 末尾有思考题。

## 前置知识

- Python 基础（numpy、matplotlib）
- 信号处理入门概念（时域/频域）
- 对 EEG 有基本了解

## 预计时间

每个 Unit 约 30-60 分钟，总计约 4-6 小时。
