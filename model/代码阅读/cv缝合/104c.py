import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import font_manager

# ✅ 1. 设置中文字体（路径需根据系统调整）
font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows示例，Mac/Linux路径不同
my_font = font_manager.FontProperties(fname=font_path)

# 全局字体配置
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ✅ 2. 字段列表
fields = ["大洲", "国家", "纬度", "经度", "海拔", "平均深度", "最大深度", "表面积", "入流溪流",
          "流域面积", "州", "发电厂排放", "冰期", "是否结冰"]

# ✅ 3. 生成更真实的相关系数矩阵（对称且半正定）
def generate_correlation_matrix(n, seed=42):
    np.random.seed(seed)
    # 生成随机数据矩阵
    data = np.random.randn(n, n)
    # 计算协方差矩阵
    cov_matrix = np.cov(data)
    # 转换为相关系数矩阵
    diag = np.sqrt(np.diag(cov_matrix))
    corr = cov_matrix / np.outer(diag, diag)
    # 限制数值范围在[-0.7, 0.7]之间（避免过高的随机相关性）
    corr = np.clip(corr, -0.7, 0.7)
    # 强制对称（确保矩阵严格对称）
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    print(corr)

    return corr

n = len(fields)
corr = generate_correlation_matrix(n)

# ✅ 4. 手动设置与“是否结冰”高相关性的字段（覆盖随机生成的值）
target_idx = fields.index("是否结冰")
high_corr = {"纬度": 0.85, "海拔": 0.65, "冰期": 0.70, "国家": 0.55, "州": 0.60}
for feat, val in high_corr.items():
    idx = fields.index(feat)
    corr[idx, target_idx] = val
    corr[target_idx, idx] = val  # 确保对称性

# ✅ 5. 构建DataFrame
df_corr = pd.DataFrame(corr, index=fields, columns=fields)

# ✅ 6. 绘制热力图（优化注释显示）
fig, ax = plt.subplots(figsize=(16, 14))  # 增大画布尺寸
heatmap = sns.heatmap(
    df_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    annot_kws={
        "size": 10,
        "family": "DejaVu Sans",  # 确保数字字体兼容
        "color": "black"           # 强制黑色字体
    },
    cbar_kws={"shrink": 0.8},
    ax=ax
)

# ✅ 7. 调整标签样式
ax.set_title("字段相关性热力图（是否结冰相关性突出）", fontsize=18, fontproperties=my_font)
plt.xticks(
    rotation=45,
    ha='right',
    fontsize=12,
    fontproperties=my_font
)
plt.yticks(
    rotation=0,
    fontsize=12,
    fontproperties=my_font
)

# ✅ 8. 优化布局
plt.tight_layout(pad=3.0)
plt.show()