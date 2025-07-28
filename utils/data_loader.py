"""
数据加载模块：负责加载和预处理数据
"""
import pandas as pd

class DataLoader:
    def __init__(self, factor_path, return_path):
        """
        初始化数据加载器
        :param factor_path: 因子数据路径
        :param return_path: 收益率数据路径
        """
        self.factor_path = factor_path
        self.return_path = return_path

    
    def load_and_merge_data(self):
        """
        加载并合并因子和收益率数据
        :return: 合并后的DataFrame
        """
        # 加载数据
        factor_df = pd.read_csv(self.factor_path, parse_dates=['datetime'])
        return_df = pd.read_csv(self.return_path, parse_dates=['datetime'])
        
        # 合并数据
        data = pd.merge(factor_df, return_df, on=['datetime', 'sec_code'], how='left')
        data.rename(columns={'ret': 'target'}, inplace=True)
        
        return data
    
    def split_train_predict_data(self, data):
        """
        分割训练数据和预测数据
        :param data: 合并后的完整数据
        :return: (训练数据, 预测数据)
        """
        max_date = data['datetime'].max()
        train_data = data[data['datetime'] < max_date].copy()
        predict_data = data[data['datetime'] == max_date].copy()
        return train_data, predict_data
    
    def get_monthly_splits(self, data, train_window=12):
        """
        生成按月滚动的训练测试分割
        :param data: 训练数据
        :param train_window: 训练窗口大小(月)
        :return: 分割后的数据迭代器
        """
        data = data.sort_values(['datetime', 'sec_code'])
        months = sorted(data['datetime'].unique())
        
        for i in range(train_window, len(months)):
            train_months = months[i-train_window:i]
            test_month = months[i]
            train = data[data['datetime'].isin(train_months)].copy()
            test = data[data['datetime'] == test_month].copy()
            yield train, test