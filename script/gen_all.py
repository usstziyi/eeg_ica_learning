"""生成全部 6 个 ICA 学习 notebook（修复版）"""
import json, os

OUT = os.path.join(os.path.dirname(__file__), "notebooks")
os.makedirs(OUT, exist_ok=True)

def nb(cells):
    return json.dumps({"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10.0"}},"nbformat":4,"nbformat_minor":5}, indent=1, ensure_ascii=False)

def M(s): return {"cell_type":"markdown","metadata":{},"source":[l+"\n" for l in s.split("\n")]}
def C(s): return {"cell_type":"code","metadata":{},"source":[l+"\n" for l in s.split("\n")],"outputs":[],"execution_count":None}

# ==================== Unit 0 ====================
u0 = [
M("""# Unit 0：环境准备

## 目标
- 安装 MNE-Python
- 下载 mne-sample-data 数据集
- 验证环境可用"""),
M("""### 0.1 安装 MNE

在终端运行（二选一）：
```bash
pip install mne scikit-learn
# 或 conda install -c conda-forge mne scikit-learn
```"""),
C("""import mne, numpy as np, matplotlib.pyplot as plt, os
print(f"MNE 版本: {mne.__version__}")
print(f"NumPy 版本: {np.__version__}")
print("✅ 环境检查通过！")"""),
M("""### 0.2 下载 sample data

首次运行会下载约 1.5GB 数据。"""),
C("""sample_data_dir = mne.datasets.sample.data_path()
print(f"Sample data 路径: {sample_data_dir}")
# 设置 EEG 通道的 montage（电极位置）
montage = mne.channels.make_standard_montage('standard_1020')
print(f"Montage 通道数: {len(montage.ch_names)}")"""),
M("""### 0.3 预览原始数据

> ⚠️ 注意：本数据集的 EEG 通道命名为 `EEG 001`、`EEG 002`... 而非标准的 Fp1、Cz 等。
> 这是 MNE sample data 的历史命名方式，不影响使用。"""),
C("""raw_fname = sample_data_dir / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=False)
print(raw)
print(f"采样率: {raw.info['sfreq']} Hz  |  通道数: {len(raw.ch_names)}  |  时长: {raw.times[-1]:.0f}s")
# 只看 EEG 通道
raw_eeg = raw.copy().pick(['eeg'])
print(f"EEG 通道（前5个）: {raw_eeg.ch_names[:5]}...")"""),
C("""# 可视化前 60 秒
raw_eeg.crop(0, 60).plot(n_channels=10, scalings='auto', title='EEG 原始数据 (60s)')"""),
M("""### ✅ Unit 0 完成！

继续 → **Unit 1：ICA 数学直觉**"""),
]

# ==================== Unit 1 ====================
u1 = [
M("""# Unit 1：ICA 的数学直觉 —— 鸡尾酒会问题

## 目标
- 理解「盲源分离」概念
- 用代码模拟鸡尾酒会问题
- 直观感受 ICA 如何从混合信号中恢复源信号

## 核心问题
> 房间里三个人同时说话，两个麦克风录下混合信号。
> **只凭混合录音，能否还原每个人的原始语音？**"""),
M("""### 1.1 生成模拟"源信号"

我们用三种不同的信号模拟"说话者"：
- 正弦波（100 Hz，光滑振荡）
- 正弦波（300 Hz，不同相位）
- 方波（500 Hz，陡峭切换）"""),
C("""import numpy as np
import matplotlib.pyplot as plt

# 参数
fs = 2000           # 采样率 2000 Hz
t = np.arange(0, 1, 1/fs)  # 1 秒
n_samples = len(t)

# 三个「说话者」——三种不同的源信号
s1 = np.sin(2 * np.pi * 100 * t)              # 100 Hz 正弦波
s2 = np.sin(2 * np.pi * 300 * t + 1.5)        # 300 Hz 正弦波（相位偏移）
s3 = np.sign(np.sin(2 * np.pi * 500 * t))     # 500 Hz 方波

S = np.vstack([s1, s2, s3])  # shape: (3, n_samples)

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
labels = ['源信号 1 (100 Hz 正弦)', '源信号 2 (300 Hz 正弦)', '源信号 3 (500 Hz 方波)']
for ax, s, label in zip(axes, S, labels):
    ax.plot(t[:200], s[:200], linewidth=0.8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xlim(0, 0.1)
axes[-1].set_xlabel('时间 (s)')
fig.suptitle('源信号（三个独立信号源）', fontsize=14)
plt.tight_layout()
plt.show()"""),
M("""### 1.2 混合信号 —— 模拟麦克风录音

每个麦克风收到的是三个源信号的**线性加权和**：

$$X = A \\cdot S$$

其中 $A$ 是混合矩阵（未知），$X$ 是观测信号（已知）。"""),
C("""# 随机混合矩阵（模拟麦克风位置不同导致的权重差异）
np.random.seed(42)
A = np.random.randn(3, 3)
X = A @ S  # (3, 3) @ (3, n_samples) = (3, n_samples)

print("混合矩阵 A（在真实问题中未知！）:")
print(np.array2string(A, precision=3, suppress_small=True))

# 可视化混合信号
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for ax, x, label in zip(axes, X, ['麦克风 1', '麦克风 2', '麦克风 3']):
    ax.plot(t[:200], x[:200], linewidth=0.8, color='crimson')
    ax.set_ylabel(label)
    ax.set_xlim(0, 0.1)
axes[-1].set_xlabel('时间 (s)')
fig.suptitle('观测信号（麦克风录到的混合信号）—— 已看不出原始形态', fontsize=14)
plt.tight_layout()
plt.show()"""),
M("""### 1.3 ICA 的核心假设

| 假设 | 含义 |
|------|------|
| **统计独立** | 源信号彼此独立（说话者互不影响） |
| **非高斯性** | 源信号最多一个高斯分布（中心极限定理保证混合信号更高斯） |
| **线性混合** | $X = AS$，瞬时线性混合 |
| **源数 ≤ 观测数** | 麦克风数量 ≥ 说话者数量 |

### 1.4 从混合信号恢复源信号"""),
C("""from sklearn.decomposition import FastICA

# FastICA 尝试从 X 中恢复独立成分
ica = FastICA(n_components=3, random_state=42, max_iter=2000)
S_hat = ica.fit_transform(X.T).T  # 恢复的源信号

# 可视化对比：原始 vs 恢复
fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex=True, sharey='row')
for i in range(3):
    axes[i, 0].plot(t[:200], S[i, :200], linewidth=0.8)
    axes[i, 0].set_ylabel(f'源 {i+1}')
    axes[i, 1].plot(t[:200], S_hat[i, :200], linewidth=0.8, color='green')
    axes[i, 1].set_ylabel(f'恢复 {i+1}')
axes[0, 0].set_title('原始源信号')
axes[0, 1].set_title('ICA 恢复的信号')
axes[-1, 0].set_xlabel('时间 (s)')
axes[-1, 1].set_xlabel('时间 (s)')
fig.suptitle('ICA 盲源分离效果对比', fontsize=14)
plt.tight_layout()
plt.show()

print("✅ 注意：ICA 恢复的信号可能有符号翻转和幅值缩放，但波形形态被完美保留！")"""),
M("""### 1.5 相关性验证

用 Pearson 相关系数量化恢复效果（绝对值越大越好）。"""),
C("""from scipy.stats import pearsonr

print("源信号 vs 恢复信号 的相关性矩阵:")
print("        恢复1     恢复2     恢复3")
for i in range(3):
    corrs = [pearsonr(S[i], S_hat[j])[0] for j in range(3)]
    print(f"源{i+1}  {corrs[0]:>8.3f}  {corrs[1]:>8.3f}  {corrs[2]:>8.3f}")

# 每行最大值应该接近 1 或 -1
print("\\n✅ 每行都有一个恢复信号与之高度相关 → ICA 成功分离！")
print("   注意符号（±）无所谓，波形形态才是关键。")"""),
M("""### 1.6 ICA 的两种固有模糊性

ICA 存在两个不可消除的不确定性（但在 EEG 中都不是问题）：

1. **幅值模糊**：恢复信号的幅值可能被缩放（$s$ vs $k \\cdot s$）
   → EEG 中我们只关心波形形态，不要求绝对电压

2. **顺序模糊**：成分的排列顺序是随机的
   → 我们通过地形图/时间序列来「识别」成分，不依赖顺序

### 💡 关键理解

1. **ICA 不需要知道混合矩阵 A** —— 这就是「盲」源分离
2. **ICA 利用的是统计独立性** —— 不是频率差异、不是空间位置
3. **恢复的信号形态正确，但幅值和顺序可能变化**

### 🤔 思考题

- 如果两个源信号都是纯正弦波，ICA 还能分离吗？试试看。
- 如果麦克风数 < 说话者数（欠定问题），会发生什么？

→ 进入 **Unit 2：ICA 算法原理**"""),
]

# ==================== Unit 2 ====================
u2 = [
M("""# Unit 2：ICA 算法原理 —— FastICA 详解

## 目标
- 理解 ICA 的目标函数：非高斯性最大化
- 掌握 FastICA 的迭代过程
- 理解白化（whitening）的作用"""),
M("""### 2.1 为什么「非高斯性」是关键？

**中心极限定理**：多个独立随机变量之和的分布，比任何一个单变量的分布**更高斯**。

混合信号 $X = AS$ 是多个源信号的加权和 → 比任何单源信号都更「高斯」。

所以 ICA 的思路是：**找到让信号「最非高斯」的方向 → 那就是源信号的方向。**"""),
C("""import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

np.random.seed(42)
n = 10000

# 三种分布：均匀、拉普拉斯、高斯
x_uniform = np.random.uniform(-1, 1, n)
x_laplace = np.random.laplace(0, 1, n)
x_gaussian = np.random.randn(n)

# 混合后（中心极限定理的体现）
x_mix = (x_uniform + x_laplace) / 2

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, data, title in zip(axes.flat,
    [x_uniform, x_laplace, x_gaussian, x_mix],
    ['均匀分布', '拉普拉斯分布', '高斯分布', '混合信号 (均匀+拉普拉斯)']):
    ax.hist(data, bins=80, density=True, alpha=0.7, color='steelblue')
    ax.set_title(f'{title}\\n峰度(Kurtosis) = {kurtosis(data):.2f}')
plt.suptitle('不同分布的峰度对比（高斯=0，非高斯偏离0）', fontsize=14)
plt.tight_layout()
plt.show()

print("关键：混合信号峰度 → 0（更高斯），ICA 通过最大化非高斯性分离信号")"""),
M("""### 2.2 数据预处理之一：中心化

减去均值，让数据零中心，简化后续计算。"""),
C("""# 沿用 Unit 1 的模拟数据
fs = 2000; t = np.arange(0, 1, 1/fs)
s1 = np.sin(2 * np.pi * 100 * t)
s2 = np.sin(2 * np.pi * 300 * t + 1.5)
s3 = np.sign(np.sin(2 * np.pi * 500 * t))
S = np.vstack([s1, s2, s3])

np.random.seed(42)
A = np.random.randn(3, 3)
X = A @ S

X_centered = X - X.mean(axis=1, keepdims=True)
print(f"中心化前均值: {np.array2string(X.mean(axis=1), precision=6)}")
print(f"中心化后均值: {np.array2string(X_centered.mean(axis=1), precision=6)}")
print("→ 均值归零 ✓")"""),
M("""### 2.3 数据预处理之二：白化（Whitening）

白化 = 去相关 + 方差归一化。变换后满足：
$$E[Z Z^T] = I$$

即各通道方差为 1，互协方差为 0。

**为什么要白化？** 它把 ICA 的搜索空间从一个任意矩阵降为**正交矩阵**，大幅简化问题。"""),
C("""def whiten(X):
    \"\"\"手工实现白化，帮助理解原理\"\"\"
    cov = np.cov(X)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 白化矩阵 = D^(-1/2) @ E^T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + 1e-10))
    whitening_matrix = D_inv_sqrt @ eigenvectors.T
    return whitening_matrix @ X, whitening_matrix

Z, W_white = whiten(X_centered)

# 验证
cov_Z = np.cov(Z)
print("白化后协方差矩阵（应接近单位矩阵）:")
print(np.array2string(cov_Z, precision=4, suppress_small=True))

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_centered[0, :500], X_centered[1, :500], alpha=0.5, s=3)
axes[0].set_title('白化前（相关 + 方差不同）')
axes[0].set_aspect('equal')
axes[1].scatter(Z[0, :500], Z[1, :500], alpha=0.5, s=3, color='green')
axes[1].set_title('白化后（去相关 + 等方差 → 球状分布）')
axes[1].set_aspect('equal')
fig.suptitle('白化使数据球化', fontsize=14)
plt.tight_layout()
plt.show()"""),
M("""### 2.4 FastICA 核心：非高斯性度量

FastICA 用**负熵近似**作为非高斯性度量：

$$J(y) \\propto [E\\{G(y)\\} - E\\{G(\\nu)\\}]^2$$

其中 $\\nu$ 是标准高斯变量，$G$ 是非二次函数。

常用 $G$ 函数：
| 函数 | $G(u)$ | 适用场景 |
|------|--------|---------|
| logcosh | $\\frac{1}{a}\\log\\cosh(au)$ | 通用（默认） |
| exp | $-\\exp(-u^2/2)$ | 超高斯（尖峰分布） |
| cube | $u^4/4$ | 基于峰度，对异常值敏感 |"""),
C("""from sklearn.decomposition import FastICA
import time

for fun in ['logcosh', 'exp', 'cube']:
    start = time.time()
    ica = FastICA(n_components=3, fun=fun, random_state=42, max_iter=2000)
    S_hat = ica.fit_transform(X.T).T
    elapsed = time.time() - start
    print(f"G={fun:8s}  迭代: {ica.n_iter_:3d}  耗时: {elapsed:.4f}s")

print("\\n默认 logcosh 兼顾速度与鲁棒性。")"""),
M("""### 2.5 FastICA 迭代过程（简化版）

核心循环（单个成分的提取）：
1. 随机初始化权重向量 $w$
2. $w^+ = E\\{Z \\cdot g(w^T Z)\\} - E\\{g'(w^T Z)\\} \\cdot w$
3. $w = w^+ / \\|w^+\\|$（归一化）
4. 与已找到的成分正交化（去相关）
5. 重复直到收敛"""),
C("""def fastica_single_component(Z, max_iter=1000, tol=1e-6):
    \"\"\"提取单个独立成分的简化 FastICA 实现\"\"\"
    n, m = Z.shape
    w = np.random.randn(n)
    w = w / np.linalg.norm(w)

    for i in range(max_iter):
        w_old = w.copy()
        # 核心不动点迭代
        wX = w @ Z
        g = np.tanh(wX)           # G' = tanh（logcosh 的导数）
        gp = 1 - np.tanh(wX)**2   # G'' = 1 - tanh²
        w_new = (Z @ g) / m - np.mean(gp) * w
        w_new = w_new / np.linalg.norm(w_new)
        w = w_new
        if abs(abs(w @ w_old) - 1) < tol:
            break
    return w, i + 1

w_ic1, iters = fastica_single_component(Z)
print(f"提取第一个独立成分，{iters} 次迭代收敛")
print(f"权重向量: {np.array2string(w_ic1, precision=4)}")"""),
M("""### 2.6 完整 ICA 流程图

```
原始数据 X
    │
    ▼
中心化 → X_centered（均值=0）
    │
    ▼
白化   → Z（协方差=I，球状分布）
    │
    ▼
FastICA 迭代
  ├─ 随机初始化 w
  ├─ 不动点迭代 w ← f(w)
  ├─ Gram-Schmidt 正交化（避免重复收敛）
  └─ 收敛判断
    │
    ▼
解混矩阵 W → 独立成分 S = W @ Z
```

### 2.7 成分数量怎么选？

在 EEG 中，成分数量 $k$ 的选择是一个权衡：

| $k$ 太少 | $k$ 太多 |
|----------|---------|
| 伪迹和脑信号混在同一成分中 | 脑信号过度拆分，出现噪声成分 |
| 剔除伪迹时误伤脑信号 | 人工识别负担增大 |

**经验法则：** 取通道数的 1/4 到 1/2。60 通道 EEG → 15-30 个成分。

### 🤔 思考题

- 不白化直接跑 ICA 会怎样？（试试跳过白化步骤）
- 为什么每次 FastICA 运行结果可能略有不同？

→ 进入 **Unit 3：EEG 伪迹识别**"""),
]

# ==================== Unit 3 ====================
u3 = [
M("""# Unit 3：EEG 中的伪迹识别

## 目标
- 认识 EEG 中的常见伪迹类型
- 理解每种伪迹的时域/频域/空间特征
- 学会肉眼 + 工具识别伪迹

> ⚠️ 本数据集的 EEG 通道名为 `EEG 001`、`EEG 002` 等数字编号。
> `EEG 001`/`EEG 002` 大致对应前额 Fp1/Fp2 位置。"""),
M("""### 3.1 加载 EEG 数据"""),
C("""import mne
import numpy as np
import matplotlib.pyplot as plt

sample_data_dir = mne.datasets.sample.data_path()
raw_fname = sample_data_dir / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick(['eeg'])
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

print(raw)
print(f"EEG 通道（前5）: {raw.ch_names[:5]}")
# 看一眼通道位置
raw.plot_sensors(show_names=True, title='EEG 电极布局')
print("\\n提示：EEG 001/002 在前额区（≈Fp1/Fp2），EEG 060 在枕区（≈Oz）")"""),
M("""### 3.2 伪迹类型一：眨眼（Eye Blink）

**特征：**
- 时域：大振幅尖峰（100-200 μV），持续 200-400 ms
- 空间：前额通道（`EEG 001`/`EEG 002`）最强，向后递减
- 频域：低频为主（< 5 Hz）"""),
C("""# 自动找出眨眼候选：在前额通道上找异常大振幅的时间点
fp1_data = raw.get_data(picks='EEG 001')[0]
fp2_data = raw.get_data(picks='EEG 002')[0]

# 找振幅超过 80 μV 的点作为眨眼候选
blink_threshold = 80e-6
blink_mask = (np.abs(fp1_data) > blink_threshold) | (np.abs(fp2_data) > blink_threshold)

# 找出连续的眨眼段
blink_events = []
in_blink = False
for i, v in enumerate(blink_mask):
    if v and not in_blink:
        start = i
        in_blink = True
    elif not v and in_blink:
        blink_events.append((start / raw.info['sfreq'], i / raw.info['sfreq']))
        in_blink = False

print(f"检测到 {len(blink_events)} 个可能的眨眼事件")
print(f"前 5 个眨眼的时间位置:")
for i, (start, end) in enumerate(blink_events[:5]):
    print(f"  #{i+1}: {start:.1f}s - {end:.1f}s (持续 {end-start:.2f}s)")"""),
C("""# 放大看第一个眨眼事件
if blink_events:
    t0 = blink_events[1][0]  # 取第二个（第一个可能在边缘）
    raw_cropped = raw.copy().crop(tmax=t0 + 3).crop(tmin=max(0, t0 - 1))
    
    # 画前 10 个通道
    fig = raw_cropped.plot(
        n_channels=20, scalings=dict(eeg=100e-6),
        title=f'眨眼伪迹示例（约 t={t0:.1f}s，注意前几个通道的大振幅脉冲）',
        duration=4
    )"""),
M("""### 3.3 伪迹类型二：眼动（Eye Movement / EOG）

**特征：**
- 水平眼动：F7/F8 位置的通道有阶梯状偏移
- 垂直眼动：Fp1/Fp2 位置的通道有缓慢漂移
- 频率：< 4 Hz，持续数百毫秒"""),
C("""# 画前额通道 vs 枕部通道对比
fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

fp1_data = raw.get_data(picks='EEG 001')[0]
axes[0].plot(raw.times[:5000], fp1_data[:5000]*1e6, linewidth=0.5)
axes[0].set_ylabel('EEG 001 (≈Fp1) [μV]')
axes[0].set_title('前额通道 — 注意低频漂移（眼动伪迹）')

# 枕部通道作为对照
oz_data = raw.get_data(picks='EEG 060')[0]
axes[1].plot(raw.times[:5000], oz_data[:5000]*1e6, linewidth=0.5, color='green')
axes[1].set_ylabel('EEG 060 (≈Oz) [μV]')
axes[1].set_title('枕部通道 — 相对干净，眼动影响小')
axes[1].set_xlabel('时间 (s)')
plt.tight_layout()
plt.show()
print("→ Fp1 有明显低频漂移（眼动），而 Oz 相对干净。这就是空间特异性。")"""),
M("""### 3.4 伪迹类型三：心电（ECG / Cardiac）

**特征：**
- 周期性尖峰，间隔 ≈ 心率周期（0.6-1.0 秒）
- 振幅 10-100 μV
- 在所有通道上出现，但振幅/极性因通道而异"""),
C("""# 在多个通道上找周期性尖峰
data_30s = raw.get_data()[:, :int(30 * raw.info['sfreq'])]
times_30s = raw.times[:int(30 * raw.info['sfreq'])]

fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
for i, ch_name in enumerate(['EEG 010', 'EEG 030', 'EEG 050']):
    ch_idx = raw.ch_names.index(ch_name)
    axes[i].plot(times_30s, data_30s[ch_idx]*1e6, linewidth=0.5)
    axes[i].set_ylabel(f'{ch_name} (μV)')
    axes[i].set_ylim(-30, 30)
axes[0].set_title('在不同通道上寻找规律出现的尖峰（心跳伪迹）')
axes[-1].set_xlabel('时间 (s)')
plt.tight_layout()
plt.show()
print("心跳伪迹线索：规律尖峰，间隔约 0.6-1s，多个通道同时出现。")"""),
M("""### 3.5 伪迹类型四：肌肉伪迹（EMG）

**特征：**
- 高频（20-300 Hz）不规则噪声
- 振幅突然增大
- 常见于颞部通道（受试者咬牙/紧张时）"""),
C("""# 对比低频段和高频段的 PSD
raw_psd = raw.compute_psd(fmin=1, fmax=100, picks=['EEG 001', 'EEG 030'])
fig = raw_psd.plot(average=True, spatial_colors=False)
plt.title('功率谱密度 — 高频段（>30Hz）异常隆起提示肌电伪迹')
plt.show()
print("肌电伪迹线索：PSD 在 20-100 Hz 的异常功率。")"""),
M("""### 3.6 伪迹总结表

| 伪迹 | 时域 | 频域 | 空间 | 振幅 |
|------|------|------|------|------|
| 眨眼 | 尖峰 200-400ms | < 5 Hz | 前额最强 | 100-200 μV |
| 眼动 | 阶梯/漂移 | < 4 Hz | F7/F8, Fp1/Fp2 | 50-100 μV |
| 心电 | 周期尖峰 ~1s | 5-20 Hz 有峰 | 全头，振幅各异 | 10-100 μV |
| 肌电 | 高频不规则 | 20-300 Hz | 颞部最强 | 变化大 |

### 🤔 思考题

- 为什么 ICA 特别适合眨眼/眼动？（它们与脑信号是独立产生的 → 统计独立）
- 肌电伪迹用 ICA 的难点？（频率范围与脑信号 β/γ 频段重叠 → 不易分离）

→ 进入 **Unit 4：MNE ICA 实战**"""),
]

# ==================== Unit 4 ====================
u4 = [
M("""# Unit 4：MNE 中 ICA 实战

## 目标
- 掌握 MNE ICA 的标准流水线
- 学会 ICA 拟合、成分可视化、识别与剔除
- 理解每个步骤的注意事项"""),
M("""### 4.1 ICA 处理流水线

```
原始 EEG
  ↓
① 滤波 (1-40 Hz 带通)      ← 去漂移和高频噪声，利于 ICA 收敛
  ↓
② 剔除大伪迹段 (reject)    ← 大幅值异常会绑架 ICA 成分
  ↓
③ ICA 拟合                 ← 数据干净才能得到好分解
  ↓
④ 成分可视化 & 识别         ← 地形图 + 时间序列 + 功率谱
  ↓
⑤ 剔除伪迹成分             ← apply 到原始/滤波后数据
  ↓
干净 EEG
```"""),
M("""### 4.2 步骤①：滤波

**为什么 ICA 前要滤波？**
- 高通（1 Hz）：去除 DC 漂移，漂移不是独立成分
- 低通（40 Hz）：去除肌电高频噪声，它们也不满足 ICA 的线性混合假设"""),
C("""import mne
import numpy as np
import matplotlib.pyplot as plt

sample_data_dir = mne.datasets.sample.data_path()
raw_fname = sample_data_dir / 'MEG' / 'sample' / 'sample_audvis_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick(['eeg'])
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

# 带通滤波
raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)

# 对比
fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
before = raw.get_data(picks='EEG 001')[0, :2000]
after  = raw_filtered.get_data(picks='EEG 001')[0, :2000]
times  = raw.times[:2000]
axes[0].plot(times, before*1e6, linewidth=0.5)
axes[0].set_ylabel('μV'); axes[0].set_title('滤波前（注意漂移和高频毛刺）')
axes[1].plot(times, after*1e6, linewidth=0.5, color='green')
axes[1].set_ylabel('μV'); axes[1].set_title('滤波后 (1-40 Hz)')
axes[1].set_xlabel('时间 (s)')
plt.tight_layout(); plt.show()"""),
M("""### 4.3 步骤②：剔除大伪迹段

FastICA 对大幅值异常敏感，大伪迹会把整个成分"绑架"成那个瞬间的形状。

最佳实践：用 `reject` 参数排除峰峰值过大的时间窗口。"""),
C("""# 基于峰峰值剔除：任何通道在 1 秒内峰峰值超过 200 μV 的段被排除
reject_criteria = dict(eeg=200e-6)  # 200 μV

# 创建 epochs 只是为了用 reject（不用实际上做 epoch）
events = mne.make_fixed_length_events(raw_filtered, duration=1.0)
epochs = mne.Epochs(raw_filtered, events, tmin=0, tmax=0.99,
                     baseline=None, reject=reject_criteria, preload=False)
print(f"原始 1s 段数: {len(events)}")
print(f"剔除坏段后: {len(epochs)} ({len(events)-len(epochs)} 段被排除)")
print(f"保留数据占比: {len(epochs)/len(events)*100:.1f}%")"""),
M("""### 4.4 步骤③：ICA 拟合

**成分数量怎么选？** 这里有 60 个 EEG 通道：
- 太少（如 10 个）→ 伪迹和脑信号挤在同一成分，剔除时误伤脑信号
- 太多（如 40 个）→ 脑信号过度拆分，人工识别负担大
- **推荐 15-25 个**，本教程用 15 个"""),
C("""from mne.preprocessing import ICA

n_components = 15
ica = ICA(
    n_components=n_components,
    method='fastica',
    random_state=42,
    max_iter='auto',
    fit_params=dict(extended=True)  # 扩展 FastICA，更稳定
)

print(f"开始 ICA 拟合（{n_components} 个成分，{len(raw_filtered.ch_names)} 个通道）...")
ica.fit(raw_filtered)
print(f"✅ 完成！迭代次数: {ica.n_iter_}")
print(f"解释方差比: {np.sum(ica.pca_explained_variance_[:n_components]) / np.sum(ica.pca_explained_variance_):.1%}")"""),
M("""### 4.5 步骤④：成分可视化与识别

三种互补视角来识别伪迹成分：
1. **地形图**（空间分布）—— 前额集中 → 眨眼
2. **时间序列**（时域激活）—— 间歇脉冲 → 眨眼
3. **功率谱**（频域特征）—— 低频主导 → 眼动"""),
C("""# 方式一：地形图 —— 看空间分布
ica.plot_components(picks=range(n_components))
print("观察要点：前额高亮 → 眨眼/眼动；颞部集中 → 肌电；全头分散 → 可能是心跳")"""),
C("""# 方式二：时间序列 —— 看激活模式
ica.plot_sources(raw_filtered, picks=range(min(10, n_components)))
print("观察要点：间歇性大尖峰 → 眨眼；规律脉冲 → 心跳；持续高频 → 肌电")"""),
C("""# 方式三：综合属性（地形图 + 时间序列 + 功率谱三合一）
# 先看成分 0，你自己看其他可疑成分
ica.plot_properties(raw_filtered, picks=[0])"""),
M("""### 4.6 识别伪迹成分的对照表

| 特征 | 眨眼 | 眼动 | 心跳 | 肌电 |
|------|------|------|------|------|
| 地形图 | 前额集中 🔴 | 单侧/双侧前部 | 分散或特定 | 颞部局部 |
| 时间序列 | 间歇大脉冲 | 缓慢方波/漂移 | 规律尖峰 | 持续高频噪声 |
| 功率谱 | <5 Hz 主导 | <4 Hz 主导 | 5-20 Hz 有峰 | >20 Hz 隆起 |"""),
C("""# 自动辅助：用 Fp1 找眼动相关成分
eog_indices, eog_scores = ica.find_bads_eog(
    raw, ch_name='EEG 001', threshold=3.0
)
print(f"自动检测到的眼动成分: {eog_indices}")
print(f"与 Fp1 的相关性得分: {np.array2string(eog_scores, precision=3)}")

# ⚠️ 不要盲目相信自动检测！要结合地形图和时间序列人工确认。
print("\\n💡 记住这些成分编号，和你的肉眼观察对比。Unit 5 会实际剔除。")"""),
M("""### 4.7 步骤⑤：剔除伪迹成分

```python
# 标记
ica.exclude = [0, 2]   # 根据前面的观察确定

# 应用到数据
raw_clean = ica.apply(raw_filtered.copy())

# 对比
raw_filtered.plot(n_channels=10, title='ICA 前')
raw_clean.plot(n_channels=10, title='ICA 后')
```

### ⚠️ 关键注意事项

1. **叠加因素**：先滤波再 ICA，但 `apply` 可以用在原始数据上
2. **适度剔除**：通常剔除 5-20% 的成分
3. **反复验证**：剔除后检查时间序列和地形图，确认伪迹减少
4. **随机种子**：设 `random_state` 保证可复现

### 🤔 思考题

- 为什么滤波要在 ICA 之前？（想想中心极限定理和非高斯性）
- 成分 0 是眨眼，成分 5 也是眨眼 — 为什么会分成两个成分？

→ 进入 **Unit 5：完整演练**"""),
]

# ==================== Unit 5 ====================
u5 = [
M("""# Unit 5：完整演练 —— 端到端 ICA 伪迹去除

## 目标
- 用 mne-sample-data 完成完整 ICA 流程
- 对比 ICA 前后：时域、空间、频域三维评估
- 量化去伪迹效果"""),
M("""### 5.1 加载 + 滤波 + 坏段剔除（一步到位）"""),
C("""import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# ── 加载 ──
sample_data_dir = mne.datasets.sample.data_path()
raw_fname = sample_data_dir / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.pick(['eeg'])
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)

print(f"原始: {len(raw.ch_names)} 通道, {raw.times[-1]:.0f}s, {raw.info['sfreq']} Hz")

# ── 滤波 ──
raw_filt = raw.copy().filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
raw_filt = raw_filt.notch_filter(freqs=50, fir_design='firwin')
print("✅ 滤波: 1-40 Hz 带通 + 50 Hz 陷波")

# ── 剔除大伪迹段 ──
reject_criteria = dict(eeg=200e-6)
events = mne.make_fixed_length_events(raw_filt, duration=1.0)
epochs = mne.Epochs(raw_filt, events, tmin=0, tmax=0.99,
                     baseline=None, reject=reject_criteria, preload=True)
print(f"✅ 坏段剔除: {len(events)-len(epochs)}/{len(events)} 段被排除")

# 用 epoch 重建 raw（MNE 内部会自动跳过坏段）
# 这里我们用 epoch 平均来做 ICA 拟合（epoch 已经自动 reject 了坏段）
# 或者直接用 raw 但后面 ICA fit 时传 reject
print(f"保留: {len(epochs)} 段 → 用于 ICA 拟合")"""),
M("""### 5.2 ICA 拟合"""),
C("""n_components = 20
ica = ICA(
    n_components=n_components,
    method='fastica',
    random_state=42,
    max_iter='auto',
    fit_params=dict(extended=True)
)

print(f"拟合 ICA（{n_components} 成分）...")
# 在 epochs 上拟合（自动跳过坏段）
ica.fit(epochs)
print(f"✅ 完成！迭代: {ica.n_iter_}")
print(f"解释方差比 (ICA 成分): {np.sum(ica.pca_explained_variance_[:n_components]) / np.sum(ica.pca_explained_variance_):.1%}")"""),
M("""### 5.3 成分识别 —— 三视角排查"""),
C("""# ① 地形图总览
ica.plot_components(picks=range(n_components))
print("→ 找出前额高度集中的成分（眨眼/眼动候选）")"""),
C("""# ② 前 10 个成分的时间序列
ica.plot_sources(raw_filt, picks=range(min(10, n_components)))
print("→ 找间歇脉冲（眨眼）、规律尖峰（心跳）、持续噪声（肌电）")"""),
C("""# ③ 逐个查验可疑成分
# 改成你想要细看的成分编号
suspects = [0, 1, 2]  # ← 根据前面的观察修改这个列表
for i in suspects:
    print(f"\\n{'='*40}\\n成分 {i}:")
    ica.plot_properties(raw_filt, picks=[i])"""),
M("""### 5.4 标记 + 剔除伪迹成分"""),
C("""# ── 自动检测 ──
eog_indices, eog_scores = ica.find_bads_eog(
    raw_filt, ch_name='EEG 001', threshold=2.5
)
print(f"自动检测眼动成分: {eog_indices}")

# ── 手动标记 ──
# 观察地形图 + 时间序列 + 功率谱后，把你确认的伪迹成分加进去
ica.exclude = list(eog_indices)  # 从自动检测开始

# 示例：如果你看到成分 3 也是眨眼（前额地形图 + 间歇脉冲），加上它
# ica.exclude.append(3)

print(f"\\n最终剔除: {ica.exclude}")
print(f"保留成分: {[i for i in range(n_components) if i not in ica.exclude]}")
print(f"剔除比例: {len(ica.exclude)}/{n_components} = {len(ica.exclude)/n_components:.0%}")"""),
C("""# ── 应用到数据 ──
raw_clean = ica.apply(raw_filt.copy())
print("✅ ICA 伪迹剔除完成！")"""),
M("""### 5.5 三维对比：ICA 前 vs ICA 后"""),
M("""#### ① 时域对比"""),
C("""# 前额通道 Fp1 的时间序列
ch_idx = raw_filt.ch_names.index('EEG 001')
times = raw_filt.times[:15000]
before = raw_filt.get_data()[ch_idx, :15000] * 1e6
after = raw_clean.get_data()[ch_idx, :15000] * 1e6

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)
axes[0].plot(times, before, linewidth=0.5, color='crimson', alpha=0.8)
axes[0].set_ylabel('μV')
axes[0].set_title('ICA 前 — EEG 001 (≈Fp1)，注意眨眼大脉冲')
axes[1].plot(times, after, linewidth=0.5, color='forestgreen')
axes[1].set_ylabel('μV')
axes[1].set_title('ICA 后 — EEG 001 (≈Fp1)，眨眼应被抑制')
axes[1].set_xlabel('时间 (s)')
plt.tight_layout(); plt.show()"""),
M("""#### ② 空间对比（各通道方差）"""),
C("""# 所有通道的方差：ICA 前 vs 后
var_before = raw_filt.get_data().var(axis=1)
var_after = raw_clean.get_data().var(axis=1)

fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(raw_filt.ch_names))
width = 0.35
ax.bar(x - width/2, var_before * 1e12, width, label='ICA 前',
       color='crimson', alpha=0.6)
ax.bar(x + width/2, var_after * 1e12, width, label='ICA 后',
       color='forestgreen', alpha=0.6)
ax.set_xticks(x[::3])
ax.set_xticklabels([raw_filt.ch_names[i] for i in range(0, len(x), 3)],
                   rotation=45, fontsize=8)
ax.set_ylabel('方差 (pV²)')
ax.set_title('各通道方差：红=ICA前  绿=ICA后')
ax.legend()
plt.tight_layout(); plt.show()
print("→ 前几个通道（Fp1/Fp2 区域）方差应显著下降")"""),
M("""#### ③ 频域对比（功率谱）"""),
C("""fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 前额
raw_filt.plot_psd(picks='EEG 001', fmin=1, fmax=40, ax=axes[0],
                  color='crimson', alpha=0.7, spatial_colors=False)
raw_clean.plot_psd(picks='EEG 001', fmin=1, fmax=40, ax=axes[0],
                   color='forestgreen', alpha=0.7, spatial_colors=False)
axes[0].set_title('EEG 001 (≈Fp1) 功率谱')
axes[0].legend(['ICA 前', 'ICA 后'])

# 枕部（对照）
raw_filt.plot_psd(picks='EEG 060', fmin=1, fmax=40, ax=axes[1],
                  color='crimson', alpha=0.7, spatial_colors=False)
raw_clean.plot_psd(picks='EEG 060', fmin=1, fmax=40, ax=axes[1],
                   color='forestgreen', alpha=0.7, spatial_colors=False)
axes[1].set_title('EEG 060 (≈Oz) 功率谱（对照）')
axes[1].legend(['ICA 前', 'ICA 后'])
plt.tight_layout(); plt.show()
print("→ 理想：Fp1 低频功率大幅下降，Oz 变化很小")"""),
M("""### 5.6 量化评估"""),
C("""# 计算关键通道的方差变化
fp1_idx = raw_filt.ch_names.index('EEG 001')
fp2_idx = raw_filt.ch_names.index('EEG 002')
oz_idx  = raw_filt.ch_names.index('EEG 060')

vars_before = [raw_filt.get_data()[i].var() for i in [fp1_idx, fp2_idx, oz_idx]]
vars_after  = [raw_clean.get_data()[i].var() for i in [fp1_idx, fp2_idx, oz_idx]]

print("=" * 55)
print(f"{'通道':<12} {'ICA前(pV²)':>12} {'ICA后(pV²)':>12} {'变化':>10}")
print("-" * 55)
for name, vb, va in zip(['EEG001/Fp1', 'EEG002/Fp2', 'EEG060/Oz'],
                          vars_before, vars_after):
    change = (1 - va/vb) * 100
    print(f"{name:<12} {vb*1e12:>10.1f}   {va*1e12:>10.1f}   {change:>+8.1f}%")
print("=" * 55)
print("\\n✅ 好的结果：Fp1/Fp2 方差 ↓↓（伪迹去除），Oz 基本不变（脑信号保留）")"""),
M("""### 5.7 保存结果"""),
C("""ica.save('ica_solution.fif')
print("✅ ICA 方案 → ica_solution.fif")

raw_clean.save('eeg_clean_raw.fif', overwrite=True)
print("✅ 干净 EEG → eeg_clean_raw.fif")

# 后续加载：
# ica = mne.preprocessing.read_ica('ica_solution.fif')
# raw = mne.io.read_raw_fif('eeg_clean_raw.fif')"""),
M("""### 5.8 最终总结

```
               原始 EEG (含伪迹)
                      │
                      ▼
             ① 带通滤波 1-40 Hz
                      │
                      ▼
             ② 陷波 50 Hz
                      │
                      ▼
             ③ 剔除大伪迹段 (reject >200μV)
                      │
                      ▼
             ④ ICA 拟合 (FastICA, 20 成分)
                      │
                      ▼
             ⑤ 三视角识别 (地形图 + 时序 + PSD)
                      │
                      ▼
             ⑥ 剔除伪迹成分
                      │
                      ▼
               干净 EEG ✨
```

### 🎯 你学会了什么

- [x] ICA 数学原理（鸡尾酒会 → FastICA）
- [x] EEG 常见伪迹的时-频-空三维特征
- [x] MNE ICA 完整流水线（滤波 → reject → fit → 识别 → 剔除）
- [x] 成分可视化：地形图 + 时间序列 + 功率谱
- [x] 端到端实战 + 量化评估

### 🚀 下一步

- 试试 `method='picard'`（比 FastICA 更快更稳定）
- 对比 ICA 和 SSP（信号空间投影）
- 在 ERP/时频分析前做 ICA
- 拿你自己的 EEG 数据练手

### 📚 推荐阅读

- Hyvärinen & Oja (2000): ICA: Algorithms and Applications
- MNE ICA Tutorial: https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html
- Makeig et al. (1996): ICA of Electroencephalographic Data
- Winkler et al. (2015): Automatic Classification of Artifactual ICA-Components

---

**🎉 恭喜完成全部 6 个单元！**"""),
]

# Write all
for name, cells in [
    ("unit0_environment", u0),
    ("unit1_ica_intuition", u1),
    ("unit2_fastica_theory", u2),
    ("unit3_artifact_recognition", u3),
    ("unit4_mne_ica_practice", u4),
    ("unit5_full_pipeline", u5),
]:
    path = os.path.join(OUT, f"{name}.ipynb")
    with open(path, 'w') as f:
        f.write(nb(cells))
    print(f"✅ {name}.ipynb")

print(f"\\n全部 6 个 notebook 已生成到 {OUT}/")
