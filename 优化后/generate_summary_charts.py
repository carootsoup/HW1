import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体（Windows 用 SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

# === 数据 ===
rounds = ['原始\n(基线)', '第一轮', '第二轮', '第三轮']
mse_values = [0.7592, 0.7437, 0.7188, 0.7055]
epochs = [1544, 1112, 1430, 978]
improvements = [0, 2.04, 5.32, 7.07]

loss_data = {
    '原始':    [78.85, 9.72, 3.37, 1.80, 1.08, 0.91, 0.83, 0.80, 0.77, 0.7592],
    '第一轮':  [313.31, 180.30, 28.95, 14.40, 2.84, 1.47, 1.02, 0.77, 0.74, 0.7437],
    '第二轮':  [306.83, 28.02, 9.33, 2.72, 1.40, 1.11, 0.82, 0.75, 0.73, 0.7188],
    '第三轮':  [315.43, 249.44, 249.44, 5.77, 89.39, 0.87, 0.75, 0.73, 0.720, 0.7055],
}

epoch_markers = [1, 5, 10, 20, 50, 100, 200, 500, 1000, None]

# === 图1: MSE 柱状图对比 ===
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars = ax.bar(rounds, mse_values, color=colors, edgecolor='white', linewidth=1.2)

for bar, val in zip(bars, mse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('Dev MSE Loss', fontsize=13)
ax.set_title('三轮优化 Dev MSE 对比', fontsize=15, fontweight='bold')
ax.set_ylim(0.65, 0.80)
ax.grid(axis='y', alpha=0.3)

# 添加趋势线
x_trend = np.arange(len(rounds))
ax.plot(x_trend, mse_values, 'k--', alpha=0.4, linewidth=1.5)

plt.tight_layout()
plt.savefig('summary_bar_chart.png', dpi=150)
plt.close()

# === 图2: 提升百分比 ===
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors2 = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars2 = ax.bar(rounds, improvements, color=bar_colors2, edgecolor='white', linewidth=1.2)

for bar, val in zip(bars2, improvements):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('相对基线的提升 (%)', fontsize=13)
ax.set_title('各轮优化相对基线的提升幅度', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('summary_improvement.png', dpi=150)
plt.close()

# === 图3: 训练轮数对比 ===
fig, ax = plt.subplots(figsize=(8, 5))
colors3 = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars3 = ax.bar(rounds, epochs, color=colors3, edgecolor='white', linewidth=1.2)

for bar, val in zip(bars3, epochs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=13)

ax.set_ylabel('训练 Epochs', fontsize=13)
ax.set_title('各轮训练轮数对比', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('summary_epochs.png', dpi=150)
plt.close()

# === 图4: MSE vs Epochs 散点图 ===
fig, ax1 = plt.subplots(figsize=(10, 6))

scatter_data = [
    ('原始', 0.7592, 1544),
    ('第一轮', 0.7437, 1112),
    ('第二轮', 0.7188, 1430),
    ('第三轮', 0.7055, 978),
]

cmap = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

for i, (label, mse, ep) in enumerate(scatter_data):
    ax1.scatter(ep, mse, s=300, c=cmap[i], edgecolors='black',
                linewidth=1.5, zorder=5, label=label)
    ax1.annotate(f'{label}\nMSE={mse:.4f}\nEp={ep}',
                (ep, mse),
                xytext=(15, -20 if i != 2 else 25),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color=cmap[i])

ax1.set_xlabel('训练 Epochs', fontsize=13)
ax1.set_ylabel('最佳 Dev MSE', fontsize=13)
ax1.set_title('优化路径：MSE vs 训练效率', fontsize=15, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# 添加帕累托前沿趋势
eps = [1544, 1112, 1430, 978]
mses = [0.7592, 0.7437, 0.7188, 0.7055]
ax1.plot(eps[0:2], mses[0:2], 'gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.plot(eps[2:4], mses[2:4], 'gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('summary_path.png', dpi=150)
plt.close()

# === 图5: 综合雷达图风格的对比表 ===
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

table_data = [
    ['指标', '原始（基线）', '第一轮', '第二轮', '第三轮'],
    ['网络层数', '2 层 (93→64→1)', '3 层 (128→64→1)', '4 层 (256→128→64→1)', '4 层+BN (256→128→64→1)'],
    ['激活函数', 'ReLU', 'LeakyReLU(0.1)', 'LeakyReLU(0.1)', 'LeakyReLU(0.1)'],
    ['Dropout', '无', '0.1 / 0.1', '0.2 / 0.2 / 0.1', '0.15 / 0.15 / 0.1'],
    ['优化器', 'SGD (lr=0.001)', 'Adam (lr=0.0005)', 'AdamW (lr=0.001)', 'AdamW (lr=0.001)'],
    ['正则化', '无', 'weight_decay=1e-5', 'weight_decay=1e-4', 'weight_decay=1e-4'],
    ['学习率调度', '无', 'ReduceLROnPlateau', 'CosineAnnealing(T0=100)', 'CosineAnnealing(T0=60)'],
    ['BatchNorm', '无', '无', '无', '有（每层）'],
    ['最佳 Dev MSE', '0.7592', '0.7437', '0.7188', '0.7055'],
    ['相对基线提升', '—', '↓ 2.04%', '↓ 5.32%', '↓ 7.07%'],
    ['训练 Epochs', '1544', '1112', '1430', '978'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.21, 0.21, 0.21, 0.22])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)

for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[i, j]
        if i == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif j == 0:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(fontweight='bold')
        elif j == 4:
            cell.set_facecolor('#e8f8f5')

        if i >= 9 and j >= 1:
            cell.set_facecolor('#d5f5e3' if j == 4 else cell.get_facecolor())

ax.set_title('三轮优化方案全面对比', fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('summary_comparison_table.png', dpi=150, bbox_inches='tight')
plt.close()

print("All summary charts generated successfully!")
