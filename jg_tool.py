#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/27 14:39
# @Author  : ykjiang2
# @Email   : ykjiang2@iflytek.com
# @File    : jg_tool.py
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('jg.xlsx', usecols=['时间', 'IS'], sheet_name='Sheet1')
dates = [item[0] for item in df.values]
values = [item[1]*100 for item in df.values]
# 绘制曲线图
plt.figure(figsize=(8, 5))  # 设置图表尺寸
# 设置线宽为1（细线）, 空心点(marker='o')的大小为4，并且点的边框颜色为黑色，内部为空白
plt.plot(dates, values, marker='o', c='black',markersize=4, markerfacecolor='none', markeredgewidth=1, markeredgecolor='black', linewidth=1, linestyle='-')
# 为每个数据点添加数值标签
for date, value in zip(dates, values):
    if value==0.22137900000000002:continue
    plt.text(date, value, f'{value:.3f}', ha='left', va='bottom', fontsize=8)
    # 绘制每个点的垂直辅助虚线
    plt.vlines(x=date, ymin=0.001, ymax=value, colors='gray', linestyles='--', linewidth=0.5)

# 设置y=0.1的红色横虚线
plt.axhline(y=0.1, color='red', linestyle='--', linewidth=1)
# 设置图表标题
plt.title('BCR_ABL IS')
# 显示虚线网格
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# 旋转x轴上的标签以避免重叠
plt.xticks(rotation=45)
# 设置Y轴为对数标度
plt.yscale('log')
# 定义Y轴刻度
y_ticks = [0.001, 0.01, 0.1, 1, 10, 100]
# 设置Y轴刻度值
plt.yticks(y_ticks, labels=[str(y) for y in y_ticks])
# 限制Y轴的显示范围，以使0.001成为最低刻度
plt.ylim(bottom=0.001)
# 自动调整subplot参数，使之填充整个图表区域
plt.tight_layout()
# 显示图表
plt.savefig('jg.png')