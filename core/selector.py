"""
选基策略模块：负责选择表现最好的ETF
"""
import pandas as pd

class ETFSelector:
    def __init__(self, top_n=10):
        """
        初始化选择器
        :param top_n: 选择前N只ETF
        """
        self.top_n = top_n
    
    def select_top_etfs(self, predictions_df):
        """
        选择预测收益率最高的ETF
        :param predictions_df: 包含预测结果的DataFrame
        :return: 选中的ETF DataFrame
        """
        # 按预测收益率排序并选择前N
        top_etfs = (predictions_df
                   .sort_values(['datetime', 'pred'], ascending=[True, False])
                   .groupby('datetime')
                   .head(self.top_n))
        
        # 准备输出
        output = top_etfs[['datetime', 'sec_code', 'pred', 'target']].copy()
        output.rename(columns={'pred': 'pred_return'}, inplace=True)
        output['weight'] = 1.0 / self.top_n
        
        return output
    
    def save_selection(self, selected_etfs, filepath):
        """
        保存选择结果
        :param selected_etfs: 选中的ETF DataFrame
        :param filepath: 保存路径
        """
        selected_etfs.to_csv(filepath, index=False)