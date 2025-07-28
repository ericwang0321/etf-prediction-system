"""
主程序：串联整个流程
"""
from utils.data_loader import DataLoader
from core.model import XGBoostModel, RandomForestModel, LassoModel, LSTMModel
from core.trainer import ModelTrainer
from core.evaluator import ModelEvaluator
from core.selector import ETFSelector
from utils.visualize import ResultVisualizer
import config
import os
import pandas as pd
import numpy as np

def main():
    # 初始化各模块
    data_loader = DataLoader(config.DATA_PATH['factor'], config.DATA_PATH['return'])
    evaluator = ModelEvaluator()
    selector = ETFSelector(top_n=10)
    visualizer = ResultVisualizer()
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_PATH['results'], exist_ok=True)
    os.makedirs(config.OUTPUT_PATH['models'], exist_ok=True)
    
    # 加载数据
    data = data_loader.load_and_merge_data()
    train_data, predict_data = data_loader.split_train_predict_data(data)
    
    # 【新增】: 计算 LSTM 的 input_dim，并在实例化时传入
    features_for_lstm = [col for col in train_data.columns if col not in ['datetime', 'sec_code', 'target']]
    lstm_input_dim = len(features_for_lstm)

    # 定义要测试的模型
    models = {
        'xgboost': XGBoostModel(),
        'random_forest': RandomForestModel(),
        'lasso': LassoModel(),
        'lstm': LSTMModel(input_dim=lstm_input_dim) # <-- 【修改】: 传递 input_dim
    }
    
    all_results = []
    
    # 对每个模型和训练窗口组合进行测试
    for model_name, model in models.items():
        print(f"\n===== 开始处理 {model_name} 模型 =====")
        
        # 【删除】: 移除这里对 input_dim 的特殊处理，因为它已经通过构造函数传递
        current_param_grid = config.PARAM_GRIDS[model_name]
        # if model_name == 'lstm':
        #     features = [col for col in train_data.columns if col not in ['datetime', 'sec_code', 'target']]
        #     input_dim = len(features)
        #     if isinstance(current_param_grid, dict):
        #         current_param_grid = current_param_grid.copy()
        #     current_param_grid['input_dim'] = [input_dim]
            
        # 参数调优
        trainer = ModelTrainer(model, current_param_grid)
        best_model, best_params = trainer.tune_hyperparameters(
            train_data[[col for col in train_data.columns if col not in ['datetime', 'sec_code', 'target']]],
            train_data['target']
        )
        print(f"最佳参数: {best_params}")
        
        # 测试不同训练窗口
        for window in config.TRAIN_WINDOWS:
            print(f"\n-- 测试 {window} 个月训练窗口 --")
            
            final_model, test_results = trainer.train_with_rolling_window(data_loader, window, train_data)
            
            # 评估
            metrics = evaluator.calculate_metrics(
                test_results['target'], 
                test_results['pred'],
                test_results
            )
            print(evaluator.generate_report(metrics, model_name, window))
            
            # 保存结果
            all_results.append({
                'model': model_name,
                'window': window,
                **metrics
            })
            
            # 可视化
            visualizer.plot_pred_vs_real(
                test_results, model_name, window,
                f"{config.OUTPUT_PATH['results']}/pred_vs_real_{model_name}_{window}.png"
            )
            visualizer.plot_cs_ic_series(
                test_results, model_name, window,
                f"{config.OUTPUT_PATH['results']}/cs_ic_{model_name}_{window}.png"
            )
            
            # 最终预测
            features = [col for col in predict_data.columns if col not in ['datetime', 'sec_code', 'target']]
            X_predict = predict_data[features]
            X_processed = model.preprocess_data(X_predict)
            
            final_predictions = final_model.predict(X_processed)
            if isinstance(final_predictions, np.ndarray) and final_predictions.ndim > 1:
                final_predictions = final_predictions.flatten()
            
            current_predict_data = predict_data.copy()
            current_predict_data['pred'] = final_predictions
            top_etfs = selector.select_top_etfs(current_predict_data)
            selector.save_selection(
                top_etfs,
                f"{config.OUTPUT_PATH['results']}/top_etfs_{model_name}_{window}.csv"
            )
    
    # 保存所有结果
    pd.DataFrame(all_results).to_csv(
        f"{config.OUTPUT_PATH['results']}/all_results.csv",
        index=False
    )

if __name__ == "__main__":
    main()