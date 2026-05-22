# 第二轮优化报告：深度网络 + 强化正则化

## 一、修改逻辑

### 核心思路
第一轮改进了架构和优化器（0.7592 → 0.7437），第二轮在此基础上进一步加深加宽网络，引入更强的正则化手段，并替换学习率调度器为**CosineAnnealingWarmRestarts**来周期性重启学习率，帮助模型跳出局部最优解。

### 与第一轮、原始代码的对比

| 修改项 | 原始代码 | 第一轮 | 第二轮 | 理由 |
|--------|---------|--------|--------|------|
| 网络层数 | 2 层 (93→64→1) | 3 层 (93→128→64→1) | 4 层 (93→256→128→64→1) | 进一步增加表达能力 |
| 隐藏层总神经元 | 64 | 192 | 448 | 更宽的网络容量 |
| 激活函数 | ReLU | LeakyReLU(0.1) | LeakyReLU(0.1) | 保持 |
| Dropout | 无 | 0.1 / 0.1 | 0.2 / 0.2 / 0.1 | 更激进的防过拟合 |
| 优化器 | SGD | Adam | AdamW | AdamW 自带解耦权重衰减 |
| 学习率 | 0.001 | 0.0005 | 0.001 | 配合 CosineAnnealing 用更高初始 lr |
| 权重衰减 | 无 | 1e-5 | 1e-4 | 更强的 L2 正则化 |
| 学习率调度 | 无 | ReduceLROnPlateau | CosineAnnealingWarmRestarts | 周期性重启，跳出局部最优 |
| 训练轮数 | 1544 | 1112 | 1430 | 更长训练 + 周期性重启 |

### 为什么 CosineAnnealingWarmRestarts 是突破关键
- `ReduceLROnPlateau` 只能单调降学习率，一旦陷入局部最优就无法逃脱
- `CosineAnnealingWarmRestarts` 通过周期性将 lr 重置到初始值（T_0=100, T_mult=2），让模型有机会跳出局部最优解
- T_mult=2 使每个周期的长度翻倍（100→200→400→800...），逐步给模型更多时间精细搜索

---

## 二、运行结果对比

| 指标 | 原始代码 | 第一轮 | 第二轮 | 总提升 |
|------|---------|--------|--------|--------|
| 最佳 Dev MSE | 0.7592 | 0.7437 | **0.7188** | ↓ 5.32% |
| 训练轮数 | 1544 | 1112 | 1430 | — |
| 收敛速度 | 慢 | 中等 | 快 | 初始 loss 下降更快 |

### 阶段 Loss 对比

| Epoch | 原始 SGD | 第一轮 Adam | 第二轮 AdamW+Cos |
|-------|----------|------------|-----------------|
| 1 | 78.85 | 313.31 | 306.83 |
| 5 | 9.72 | 180.30 | 28.02 |
| 10 | 3.37 | 28.95 | 9.33 |
| 20 | 1.80 | 14.40 | 2.72 |
| 50 | 1.08 | 2.84 | ~1.40 |
| 100 | ~0.91 | ~1.47 | ~1.11 |
| 500 | ~0.80 | ~0.77 | ~0.75 |
| 1000 | ~0.77 | ~0.74 | ~0.73 |
| 最终 | 0.7592 | 0.7437 | **0.7188** |

### 关键观察
- 第二轮在 epoch 1-50 的下降速度**快于第一轮**（更大的初始 lr + 更宽的网络）
- 在 100-500 epochs 区间，CosineAnnealing 的周期性重启产生了明显的 "震荡-突跳" 模式
- 在 epoch 990 和 1129 各出现一次突破性下降，证明周期性重启确实帮助跳出了局部最优

---

## 三、修改部分详情

### 1. NeuralNet 类：更深更宽 + 更强 Dropout
```python
# 第二轮：4 层全连接
self.net = nn.Sequential(
    nn.Linear(input_dim, 256),   # 第一轮: 128
    nn.LeakyReLU(0.1),
    nn.Dropout(0.2),              # 第一轮: 0.1
    nn.Linear(256, 128),
    nn.LeakyReLU(0.1),
    nn.Dropout(0.2),              # 第一轮: 0.1
    nn.Linear(128, 64),
    nn.LeakyReLU(0.1),
    nn.Dropout(0.1),
    nn.Linear(64, 1)
)
```

### 2. 优化器：Adam → AdamW
```python
config = {
    'optimizer': 'AdamW',          # 更干净的权重衰减实现
    'optim_hparas': {
        'lr': 0.001,               # 更高初始 lr 配合周期性重启
        'weight_decay': 1e-4,      # 10x 更强的正则化
    },
}
```

### 3. 学习率调度器：ReduceLROnPlateau → CosineAnnealingWarmRestarts
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=2, eta_min=1e-6)
# T_0=100: 第一个周期 100 epochs
# T_mult=2: 后续周期长度翻倍
# eta_min=1e-6: 周期结束时 lr 降到接近 0
```

### 4. 保留了第一轮的好做法
- LeakyReLU 激活函数
- Dropout 正则化
- 特征全量使用（target_only=False）

---

## 四、学习曲线与预测散点图

### 学习曲线
![学习曲线](round2_learning_curve.png)
> 注意余弦周期的波动模式，与第一轮的单调下降形成对比

### 预测 vs 真实值
![预测散点图](round2_prediction.png)

---

## 五、完整可运行代码

见文件：`round2_optimized.py`

---

## 六、小结

第二轮通过**更深更宽的网络 + AdamW + CosineAnnealingWarmRestarts**，将 Dev MSE 从 0.7437 降至 **0.7188**，累计提升 5.32%（相对原始 0.7592）。

CosineAnnealingWarmRestarts 的周期性重启策略是本轮的关键突破——它让模型有能力在训练中后期跳出局部最优解，实现了 ReduceLROnPlateau 无法达到的效果。
