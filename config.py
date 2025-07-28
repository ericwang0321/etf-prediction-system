"""
配置文件：集中管理所有路径和参数配置
"""
# 数据路径配置
DATA_PATH = {
    'factor': 'data/factor_daily.csv',    # 因子数据文件
    'return': 'data/return_monthly.csv'   # 收益率数据文件
}

# 输出路径配置
OUTPUT_PATH = {
    'results': './outputs/results',  # 结果输出目录
    'models': './outputs/models'     # 模型保存目录
}

# 模型参数网格配置
PARAM_GRIDS = {
    'xgboost': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    },
    'random_forest': {
        'n_estimators': [50, 100],
        'max_depth': [3, 5]
    },
    'lasso': {
        'alpha': [0.0001, 0.0005, 0.001]
    },
    'lstm': {
        # 【关键修改】: 为模型架构参数添加 'model__' 前缀
        'model__units': [32, 64],
        'model__dropout_rate': [0.1, 0.3],
        'model__learning_rate': [0.001, 0.005], # 这是 optimizer 的学习率
        
        # 这些是 KerasRegressor 自身的参数，无需前缀
        'epochs': [5, 10],
        'batch_size': [16, 32],
        'verbose': [0]
    }
}

# 训练窗口配置
TRAIN_WINDOWS = [12, 24, 36]  # 测试的训练窗口大小(月)