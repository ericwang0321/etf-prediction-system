# ETF收益率预测与选基系统

基于机器学习的量化策略系统，用于预测ETF未来收益率并选择表现最佳的ETF组合。

## 功能特性

- 支持多种机器学习模型：XGBoost、随机森林、Lasso回归、LSTM神经网络
- 完整的量化策略流程：数据加载、特征工程、模型训练、回测评估、选基策略
- 丰富的可视化功能：预测vs真实值对比、横截面IC时间序列
- 参数调优：网格搜索自动寻找最优超参数
- 模块化设计：符合OOP原则，易于扩展和维护

## 安装指南

### 前置要求

- Python 3.8+

### 使用说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/etf-prediction-system.git
cd etf-prediction-system
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置参数（可选）：
   修改`config.py`中的参数设置，包括：
   - 数据路径
   - 模型参数网格
   - 训练窗口设置

5. 运行主程序：
```bash
python main.py
```

### 输出结果

程序运行完成后，将在`outputs/`文件夹中生成：

- `results/` - 包含：
  - 各模型评估指标 (`all_results.csv`)
  - 预测与真实值对比图 (`pred_vs_real_*.png`)
  - IC时间序列图 (`cs_ic_*.png`)
  - 选中的ETF列表 (`top_etfs_*.csv`)
  
- `models/` - 保存的训练好的模型（可选）

### 自定义配置

可以通过修改`config.py`文件来自定义：

```python
# 示例：修改XGBoost参数网格
PARAM_GRIDS = {
    'xgboost': {
        'n_estimators': [50, 100, 150],  # 增加更多选项
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    # ...其他模型配置
}

# 示例：修改训练窗口
TRAIN_WINDOWS = [6, 12, 24]  # 测试6个月、12个月和24个月的窗口
```

## 项目结构

```
etf-prediction-system/
├── config.py             # 配置文件
├── main.py               # 主程序入口
├── requirements.txt      # 依赖列表
├── data/                 # 数据目录
│   ├── factor_daily.csv  # 因子数据
│   └── return_monthly.csv# 收益率数据
├── core/                 # 核心模块
│   ├── model.py          # 模型实现
│   ├── trainer.py        # 训练逻辑
│   ├── evaluator.py      # 评估指标
│   └── selector.py       # 选基策略
├── utils/                # 工具模块
│   ├── data_loader.py    # 数据加载
│   └── visualize.py      # 可视化
└── outputs/              # 输出目录（自动生成）
    ├── results/          # 结果文件
    └── models/           # 保存的模型
```

## 模型说明

系统支持以下机器学习模型：

| 模型名称 | 描述 | 适用场景 |
|----------|------|----------|
| XGBoost | 梯度提升决策树 | 处理非线性关系，抗过拟合 |
| 随机森林 | 集成学习方法 | 稳健预测，处理高维特征 |
| Lasso回归 | 线性模型+L1正则化 | 特征选择，解释性强 |
| LSTM | 长短期记忆网络 | 捕捉时序依赖关系 |

## 常见问题

### 如何添加新模型？

1. 在`core/model.py`中创建新模型类，继承`BaseModel`
2. 实现`build_model()`和`preprocess_data()`方法
3. 在`config.py`的`PARAM_GRIDS`中添加参数网格
4. 在`main.py`的`models`字典中添加新模型实例

### 如何修改评估指标？

编辑`core/evaluator.py`中的`calculate_metrics()`方法，添加或修改指标计算逻辑。

### 结果可视化不显示中文？

确保系统已安装中文字体（如SimHei），或在`visualize.py`中修改字体配置。
