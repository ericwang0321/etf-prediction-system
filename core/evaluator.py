"""
评估模块：负责模型性能评估
"""
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, df):
        """
        计算评估指标
        :param y_true: 真实值
        :param y_pred: 预测值
        :param df: 包含日期和证券代码的完整DataFrame
        :return: 指标字典
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'IC': spearmanr(y_true, y_pred)[0]
        }
        
        # 计算横截面IC
        cs_ic_list = []
        for dt, group in df.groupby('datetime'):
            if len(group) >= 5:
                ic = spearmanr(group['target'], group['pred'])[0]
                if not np.isnan(ic):
                    cs_ic_list.append(ic)
        metrics['CS-IC'] = np.mean(cs_ic_list) if cs_ic_list else np.nan
        
        return metrics
    
    @staticmethod
    def generate_report(metrics, model_name, train_window):
        """
        生成评估报告
        :param metrics: 指标字典
        :param model_name: 模型名称
        :param train_window: 训练窗口
        :return: 报告字符串
        """
        report = f"\n===== {model_name} 模型 ({train_window}个月窗口) 评估结果 =====\n"
        for k, v in metrics.items():
            report += f"{k}: {v:.4f}\n"
        return report