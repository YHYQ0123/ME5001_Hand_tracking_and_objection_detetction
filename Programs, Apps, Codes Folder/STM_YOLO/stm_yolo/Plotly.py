import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as mtick

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

# 阶段和指标名
stages = [f"Stage {i}" for i in range(5)]
metrics = {
    "Accuracy": "Accuracy (%)",
    "Precision": "Precision (%)",
    "Recall": "Recall (%)",
    "F1": "F1 Score (%)"
}

# 初始化数据容器
baseline_data = {m: [] for m in metrics}
stm_data = {m: [] for m in metrics}
valid_indices = set(range(5))  # 记录有效 stage 索引

# 设置路径
path = "./"  # 修改为你的文件路径

# 读取数据
for i in range(5):
    baseline_file = os.path.join(path, f"baseline_stage_{i}_metrics.csv")
    stm_file = os.path.join(path, f"stm_stage_{i}_metrics.csv")
    try:
        baseline_df = pd.read_csv(baseline_file)
        stm_df = pd.read_csv(stm_file)

        for m, csv_key in metrics.items():
            base_val = baseline_df.loc[baseline_df['Metric'] == csv_key, 'Value'].values
            stm_val = stm_df.loc[stm_df['Metric'] == csv_key, 'Value'].values

            # 如果缺失数据就跳过这个阶段
            if len(base_val) == 0 or len(stm_val) == 0:
                print(f"⚠️ Missing data in Stage {i}, metric: {m}")
                valid_indices.discard(i)
                break

            baseline_data[m].append(float(base_val[0]))
            stm_data[m].append(float(stm_val[0]))

    except Exception as e:
        print(f"❌ Error reading Stage {i}: {e}")
        valid_indices.discard(i)

# 只保留有效阶段
valid_indices = sorted(list(valid_indices))
valid_stages = [f"Stage {i}" for i in valid_indices]
x = np.arange(len(valid_stages))
bar_width = 0.35

# 2x2 子图绘图
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()
colors = ["#8884d8", "#82ca9d"]

for idx, m in enumerate(metrics):
    ax = axs[idx]
    base_vals = [baseline_data[m][i] for i in range(len(valid_indices))]
    stm_vals = [stm_data[m][i] for i in range(len(valid_indices))]

    ax.bar(x - bar_width / 2, base_vals, width=bar_width, label='Baseline', color=colors[0])
    ax.bar(x + bar_width / 2, stm_vals, width=bar_width, label='STM-YOLO', color=colors[1])
    ax.set_title(f"{m} Comparison", fontweight='bold')
    ax.set_xlabel("Assembly Stage", fontweight='bold')
    ax.set_ylabel(f"{m} (%)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_stages)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend()

plt.tight_layout()
plt.savefig("stm_vs_baseline_metrics_subplots.png", dpi=300)
plt.show()

