以下是一个适合放在GitHub上的README.md文件内容，包含了项目介绍、安装指南、使用说明和示例等内容：

```markdown
# ETF收益率预测与选基系统

基于机器学习的量化策略系统，用于预测ETF未来收益率并选择表现最佳的ETF组合。

## 功能特性

- 🚀 支持多种机器学习模型：XGBoost、随机森林、Lasso回归、LSTM神经网络
- 📊 完整的量化策略流程：数据加载、特征工程、模型训练、回测评估、选基策略
- 📈 丰富的可视化功能：预测vs真实值对比、横截面IC时间序列
- ⚙️ 参数调优：网格搜索自动寻找最优超参数
- 📂 模块化设计：符合OOP原则，易于扩展和维护

## 安装指南

### 前置要求

- Python 3.8+
- pip

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/etf-prediction-system.git
cd etf-prediction-system
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

将以下数据文件放在项目根目录下的`data/`文件夹中：

- `factor_daily.csv` - 每日因子数据（需包含datetime, sec_code列）
- `return_monthly.csv` - 月度收益率数据（需包含datetime, sec_code, ret列）

数据格式示例：
```
datetime,sec_code,factor1,factor2,factor3
2020-01-01,510300,0.12,-0.05,1.23
2020-01-01,510500,-0.08,0.15,0.87
...
```

## 使用说明

### 快速开始

1. 配置参数（可选）：
   修改`config.py`中的参数设置，包括：
   - 数据路径
   - 模型参数网格
   - 训练窗口设置

2. 运行主程序：
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
├── data/                 # 数据目录（需自行创建）
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

## 贡献指南

欢迎提交Issue或Pull Request！贡献前请阅读：
1. Fork仓库并创建特性分支
2. 提交代码前运行测试
3. 确保代码符合PEP8规范
4. 更新相关文档

## 许可证

本项目采用 MIT 许可证 - 详情参见[LICENSE](LICENSE)文件
```

这个README文件包含了用户需要了解的所有关键信息，包括：
1. 项目简介和特性
2. 安装和配置指南
3. 使用说明和示例
4. 项目结构说明
5. 模型和技术细节
6. 常见问题解答
7. 贡献指南

你可以根据实际项目情况调整内容，比如添加更多示例、截图或演示视频链接等。对于GitHub项目，一个好的README应该让用户能够快速理解项目用途并顺利开始使用。