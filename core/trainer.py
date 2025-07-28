"""
训练模块：负责模型训练和参数调优
"""
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd # <-- 确保 pandas 已导入，因为 Trainer 中使用了它

class ModelTrainer:
    def __init__(self, model, param_grid):
        """
        初始化训练器
        :param model: 模型实例
        :param param_grid: 参数网格
        """
        self.model = model
        self.param_grid = param_grid
        self.best_model = None
    
    def tune_hyperparameters(self, X, y):
        """
        执行网格搜索调参
        :param X: 特征数据
        :param y: 目标值
        :return: 最佳模型和参数
        """
        # 预处理数据
        X_processed = self.model.preprocess_data(X)
        
        # 初始化网格搜索
        grid_search = GridSearchCV(
            estimator=self.model.build_model({}),
            param_grid=self.param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # 执行搜索
        grid_search.fit(X_processed, y)
        
        # 保存最佳模型
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_
    
    # 【关键修改】: 方法签名新增 train_data_df 参数
    def train_with_rolling_window(self, data_loader, train_window, train_data_df): 
        """
        滚动窗口训练
        :param data_loader: 数据加载器实例
        :param train_window: 训练窗口大小
        :param train_data_df: 训练数据DataFrame
        :return: 测试集预测结果
        """
        all_test_results = pd.DataFrame()
        
        # 【关键修改】: 调用 get_monthly_splits 时传入 train_data_df
        for train_df, test_df in data_loader.get_monthly_splits(train_data_df, train_window):
            # 准备数据
            features = [col for col in train_df.columns if col not in ['datetime', 'sec_code', 'target']]
            X_train = train_df[features]
            y_train = train_df['target']
            X_test = test_df[features]
            
            # 预处理数据
            X_train_processed = self.model.preprocess_data(X_train)
            X_test_processed = self.model.preprocess_data(X_test)
            
            # 训练模型
            self.best_model.fit(X_train_processed, y_train)
            
            # 预测
            y_pred = self.best_model.predict(X_test_processed)
            if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            
            # 保存结果
            test_df = test_df.copy()
            test_df['pred'] = y_pred
            all_test_results = pd.concat([all_test_results, test_df])
        
        return self.best_model, all_test_results