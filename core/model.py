"""
模型模块：定义所有模型类和基类
"""
from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
import numpy as np 
import pandas as pd 

class BaseModel(ABC):
    """模型抽象基类"""
    def __init__(self, name):
        self.name = name
        self.model = None
    
    @abstractmethod
    def build_model(self, params):
        """构建模型"""
        pass
    
    @abstractmethod
    def preprocess_data(self, X):
        """数据预处理"""
        pass

class XGBoostModel(BaseModel):
    """XGBoost模型实现"""
    def __init__(self):
        super().__init__('xgboost')
        
    def build_model(self, params):
        self.model = XGBRegressor(**params)
        return self.model
    
    def preprocess_data(self, X):
        return X  # XGBoost不需要特殊预处理

class LassoModel(BaseModel):
    """Lasso回归模型实现"""
    def __init__(self):
        super().__init__('lasso')
        
    def build_model(self, params):
        self.model = Lasso(**params)
        return self.model
    
    def preprocess_data(self, X):
        return X  # Lasso不需要特殊预处理

class RandomForestModel(BaseModel):
    """随机森林模型实现"""
    def __init__(self):
        super().__init__('random_forest')
        
    def build_model(self, params):
        self.model = RandomForestRegressor(**params)
        return self.model
    
    def preprocess_data(self, X):
        return X  # 随机森林不需要特殊预处理

class LSTMModel(BaseModel):
    """LSTM模型实现"""
    def __init__(self, input_dim=None):
        super().__init__('lstm')
        self.input_dim = input_dim
        
    # 【关键修改】: build_model 方法现在只处理 KerasRegressor 自身的参数，并返回一个可调用的模型构建函数
    def build_model(self, params=None): 
        if params is None:
            params = {}

        if self.input_dim is None:
            raise ValueError("LSTMModel.build_model: 'input_dim' is not set. It must be passed to LSTMModel constructor.")

        # 【关键修改】: 定义 Keras 模型架构构建函数，它接收其自身的参数
        # 这些参数会由 GridSearchCV 通过 'model__' 前缀来传递
        def create_lstm_architecture(units=32, dropout_rate=0.1, learning_rate=0.001):
            model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(units, return_sequences=False, 
                                   input_shape=(1, self.input_dim)), 
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(1)
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            return model
        
        self.model = KerasRegressor(
            model=create_lstm_architecture, # 【关键修改】: 将模型构建函数赋值给 'model' 参数
            # KerasRegressor 自身的参数（epochs, batch_size, verbose）无需前缀
            epochs=params.get('epochs', 5),         
            batch_size=params.get('batch_size', 16),
            verbose=params.get('verbose', 0)        
        )
        return self.model
    
    def preprocess_data(self, X):
        return np.expand_dims(X, axis=1)  # LSTM需要3D输入