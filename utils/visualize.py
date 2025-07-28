"""
可视化模块：负责生成图表
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr # <-- 【新增】导入 spearmanr

# --- Matplotlib 中文显示设置 ---
import matplotlib.font_manager as fm

# 尝试查找已安装的文泉驿正黑字体
# Colab中通过 apt-get 安装的字体通常在 /usr/share/fonts/ 或 /usr/local/share/fonts/
# 优先查找 wqy-zenhei
if fm.findfont('WenQuanYi Zen Hei', fontext='ttf'):
    plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
    print("Matplotlib 字体设置为 WenQuanYi Zen Hei (已找到)。")
elif fm.findfont('wqy-zenhei', fontext='ttf'): # 另一种可能的字体名
    plt.rcParams['font.family'] = ['wqy-zenhei']
    print("Matplotlib 字体设置为 wqy-zenhei (已找到)。")
else:
    # 备用方案：尝试添加常见的字体路径，如果字体文件存在但未被fm识别
    font_path_colab = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    if os.path.exists(font_path_colab): # os 模块未导入，但此处假设环境已经提供了os
        fm.fontManager.addfont(font_path_colab)
        plt.rcParams['font.family'] = ['WenQuanYi Zen Hei'] # 尝试使用这个名字
        print("Matplotlib 字体设置为 WenQuanYi Zen Hei (从路径添加)。")
        # 重新构建字体缓存可能需要重启运行时
    else:
        # 如果文泉驿字体找不到，则回退到通用无衬线字体，并给出警告
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans'] # 常用英文字体
        print("警告：文泉驿正黑字体未找到。将使用通用 sans-serif 字体。")
        print("请确保 'wqy-zenhei.ttc' 已正确安装并可访问，并考虑重启运行时。")

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# -----------------------------

class ResultVisualizer:
    @staticmethod
    def plot_pred_vs_real(df, model_name, window, save_path):
        plt.figure(figsize=(12, 6))
        plt.plot(df['datetime'], df['target'], label='真实收益率', alpha=0.7)
        plt.plot(df['datetime'], df['pred'], label='预测收益率', alpha=0.7)
        plt.title(f'{model_name} 模型 {window}个月窗口 - 预测 vs 真实收益率')
        plt.xlabel('日期')
        plt.ylabel('收益率')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # 关闭图表以释放内存

    @staticmethod
    def plot_cs_ic_series(df, model_name, window, save_path):
        # 计算月度IC
        monthly_ic = []
        for dt, group in df.groupby('datetime'):
            # 确保有足够的数据且标准差不为0，否则 spearmanr 会报错或返回 NaN
            if len(group) >= 2 and group['target'].std() > 0 and group['pred'].std() > 0:
                try:
                    ic = spearmanr(group['target'], group['pred'])[0]
                    monthly_ic.append({'datetime': dt, 'IC': ic})
                except ValueError: # 处理spearmanr可能因数据问题（如所有值相同）而抛出的ValueError
                    monthly_ic.append({'datetime': dt, 'IC': np.nan})
            else:
                monthly_ic.append({'datetime': dt, 'IC': np.nan}) # 数据不足或无方差

        ic_df = pd.DataFrame(monthly_ic).set_index('datetime').dropna()

        plt.figure(figsize=(12, 6))
        if not ic_df.empty:
            plt.plot(ic_df.index, ic_df['IC'], marker='o', linestyle='-', markersize=4)
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f'{model_name} 模型 {window}个月窗口 - 月度IC时间序列')
            plt.xlabel('日期')
            plt.ylabel('IC值')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path)
        else:
            print(f"警告: {model_name} ({window}个月窗口) 没有有效的IC数据可供绘制。")
            # 即使没有数据，也保存一个空白图，避免后续逻辑因文件缺失而报错
            plt.plot([], []) # 绘制一个空图
            plt.title(f'无有效IC数据: {model_name} ({window}个月窗口)')
            plt.savefig(save_path)
        plt.close() # 关闭图表以释放内存