# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

COVID-19 阳性病例预测的 Kaggle 竞赛项目，使用 PyTorch DNN 做回归。目标变量是 `tested_positive`（最后一天阳性率）。

## 常用命令

```bash
# 运行主程序（训练 + 预测）
python main.py

# 安装依赖（已配置 .venv）
.venv/Scripts/pip install torch numpy matplotlib
```

## 数据

- `covid.train.csv` — 2700 条训练数据，93 个特征（前 40 个为 one-hot 州编码，后 53 个为时间序列数值特征）+ 目标列
- `covid.test.csv` — 893 条测试数据，93 个特征
- `sampleSubmission.csv` — Kaggle 提交格式参考

## 核心架构

`main.py` 包含完整流水线：
- `COVID19Dataset` — 自定义 Dataset，80/20 划分（`i % 5`），per-split z-score 归一化
- `NeuralNet` — 多层全连接网络
- `train()` — 训练循环，AdamW + CosineAnnealingWarmRestarts + 早停
- `dev()` / `test()` — 验证/测试推理
- 输出 `best_submission.csv`（预测结果）、`best_learning_curve.png`、`best_prediction.png`

关键配置：batch_size=128, lr=0.001, weight_decay=1e-3, early_stop=400, T_0=80

## 优化历史

- `三轮优化/` — 前三轮优化结果（round1/2/3_optimized.py + 报告），原始基线 dev MSE ≈ 0.7055，Kaggle RMSE ≈ 1.49
- `优化后/` — 本次会话的进一步优化结果
- `optimization_summary.md` — 最终优化总结

## 注意事项

- 用户使用中文，偏好简洁回复
- 修改 `main.py` 后必须先确保能跑通，Dev MSE 必须较上一轮有提升
- 虚拟环境在 `.venv/`，已加入 `.gitignore`
- 远程仓库: `https://github.com/carootsoup/HW1`
- `要求` 文件包含用户的最新任务指令，优先阅读
