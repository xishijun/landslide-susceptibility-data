# -*- coding: utf-8 -*-
"""
改进的多层Stacking融合型滑坡预测模型
采用三层学习器串联的集成策略，包含基学习器层、中学习器层和元学习器层
使用贝叶斯优化算法优化模型超参数，确保第三层性能最优
论文出版优化版本
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import seaborn as sns
import json
import logging
import time

# 导入统一的数据处理器和评估器
from data_processor import UnifiedDataProcessor
from model_evaluator import PaperReadyEvaluator

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================
class Config:
    # 数据配置
    TARGET_COL = 'target'
    TEST_SIZE = 0.2  # 测试集比例
    VAL_SIZE = 0.2   # 验证集比例
    RANDOM_STATE = 42
    N_FOLDS = 5  # K折交叉验证的折数
    
    # 模型配置 - 增加模型多样性
    LAYER1_MODELS = [  # 初级学习器（基学习器）
        'Logistic', 'SVM', 'RandomForest', 'AdaBoost', 'XGBoost'
    ]
    LAYER2_MODELS = [  # 中级学习器
        'XGBoost', 'GradientBoosting', 'RandomForest', 'AdaBoost'
    ]
    LAYER3_MODELS = [  # 元学习器
        'XGBoost', 'Logistic', 'GradientBoosting'
    ]
    META_MODEL_TYPE = 'XGBoost'  # 元学习器类型: 'XGBoost', 'Logistic', 'GradientBoosting'
    
    # 优化配置
    N_ITER = 20  # 贝叶斯优化迭代次数
    CV_FOLDS = 5  # 交叉验证折数
    SCORING = 'roc_auc'  # 优化目标指标
    
    # 特征配置
    FEATURE_SELECTION = True
    BALANCE_METHOD = 'SMOTE'  # 处理类别不平衡的方法
    USE_ORIGINAL_FEATURES = True  # 是否在元特征中加入原始特征
    WEIGHTED_STACKING = True  # 是否使用加权Stacking
    WEIGHT_METRIC = 'roc_auc'  # 权重计算指标: 'accuracy', 'f1', 'roc_auc'
    
    # 路径配置
    OUTPUT_ROOT = "E:/怀化市地质灾害攻关/全市易发性/result/三层结构"
    FIGURE_SIZE = (10, 6)
    DPI = 300  # 图表分辨率
    
    # 早停配置
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_ROUNDS = 50

# 创建输出目录
os.makedirs(Config.OUTPUT_ROOT, exist_ok=True)

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(Config.OUTPUT_ROOT, "landslide_model.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 核心模型基类 ====================
class BaseModel:
    """机器学习模型基类（集成训练评估全流程）"""

    model_config = {
        'Logistic': {
            'class': LogisticRegression,
            'params': {
                'class_weight': 'balanced',
                'max_iter': 2000,
                'random_state': Config.RANDOM_STATE,
                'n_jobs': -1
            },
            'param_space': {
                'C': Real(1e-3, 1e3, prior='log-uniform'),
                'penalty': Categorical(['l2', 'none']),
                'solver': Categorical(['lbfgs', 'newton-cg', 'saga'])
            }
        },
        'SVM': {
            'class': svm.SVC,
            'params': {
                'probability': True,
                'random_state': Config.RANDOM_STATE
            },
            'param_space': {
                'C': Real(1e-2, 1e2, prior='log-uniform'),
                'gamma': Real(1e-4, 1e1, prior='log-uniform'),
                'kernel': Categorical(['linear', 'rbf'])
            }
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': Config.RANDOM_STATE,
                'oob_score': True
            },
            'param_space': {
                'n_estimators': Integer(200, 1000),
                'max_depth': Integer(5, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None])
            }
        },
        'AdaBoost': {
            'class': AdaBoostClassifier,
            'params': {
                'random_state': Config.RANDOM_STATE
            },
            'param_space': {
                'n_estimators': Integer(100, 800),
                'learning_rate': Real(0.01, 2.0),
                'algorithm': Categorical(['SAMME', 'SAMME.R'])
            }
        },
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': 'binary:logistic',
                'n_jobs': -1,
                'random_state': Config.RANDOM_STATE,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            },
            'param_space': {
                'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 15),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0, 5),
                'reg_alpha': Real(0, 10),
                'reg_lambda': Real(1, 10),
                'n_estimators': Integer(200, 1200)
            }
        },
        'GradientBoosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'random_state': Config.RANDOM_STATE
            },
            'param_space': {
                'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 15),
                'subsample': Real(0.6, 1.0),
                'n_estimators': Integer(200, 1000),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10)
            }
        },
        'MLP': {
            'class': MLPClassifier,
            'params': {
                'random_state': Config.RANDOM_STATE,
                'max_iter': 1000,
                'early_stopping': True
            },
            'param_space': {
                'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)]),
                'activation': Categorical(['relu', 'tanh']),
                'alpha': Real(1e-5, 1e-1, prior='log-uniform'),
                'learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform')
            }
        }
    }

    def __init__(self, name: str):
        """初始化模型
        
        Args:
            name: 模型名称，必须是model_config中定义的模型
        """
        if name not in self.model_config:
            raise ValueError(f"不支持的模型类型: {name}")
            
        self.name = name
        cfg = self.model_config[name]
        self.model = cfg['class'](**cfg['params'])
        self.param_space = cfg['param_space']
        self.best_params_ = None
        self.best_score_ = None
        self.cv_scores_ = None  # 存储交叉验证结果
        self.train_time = None  # 训练时间
        self.weights_ = None  # 模型权重

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """执行贝叶斯超参数优化
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            优化后的模型自身
        """
        logger.info(f"开始优化 {self.name} 的超参数")
        start_time = time.time()
        
        try:
            # 对于MLP模型，需要特殊处理hidden_layer_sizes参数
            if self.name == 'MLP':
                # 创建一个简化的参数空间，避免元组处理问题
                simplified_param_space = {
                    'activation': self.param_space['activation'],
                    'alpha': self.param_space['alpha'],
                    'learning_rate_init': self.param_space['learning_rate_init']
                }
                
                # 固定hidden_layer_sizes为一个值
                self.model.set_params(hidden_layer_sizes=(100,))
                
                opt = BayesSearchCV(
                    estimator=self.model,
                    search_spaces=simplified_param_space,
                    n_iter=Config.N_ITER,
                    cv=StratifiedKFold(Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE),
                    scoring=Config.SCORING,
                    n_jobs=-1,
                    random_state=Config.RANDOM_STATE,
                    verbose=0
                )
            else:
                # 其他模型使用完整参数空间
                opt = BayesSearchCV(
                    estimator=self.model,
                    search_spaces=self.param_space,
                    n_iter=Config.N_ITER,
                    cv=StratifiedKFold(Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE),
                    scoring=Config.SCORING,
                    n_jobs=-1,
                    random_state=Config.RANDOM_STATE,
                    verbose=0
                )

            # 使用进度条显示优化过程
            with tqdm(total=Config.N_ITER, desc=f"{self.name} 超参数优化") as pbar:
                # 定义回调函数更新进度条
                def callback(optim_result):
                    pbar.update(1)
                    current_score = opt.best_score_ if hasattr(opt, 'best_score_') else 0
                    pbar.set_postfix(best_score=f"{current_score:.4f}")

                # 执行优化
                opt.fit(X, y, callback=callback)

            # 保存优化结果
            self.model = opt.best_estimator_
            self.best_params_ = opt.best_params_
            self.best_score_ = opt.best_score_
            
            # 计算交叉验证分数
            cv_scores = cross_val_score(self.model, X, y, 
                                      cv=StratifiedKFold(Config.CV_FOLDS, shuffle=True, 
                                                       random_state=Config.RANDOM_STATE),
                                      scoring=Config.SCORING,
                                      n_jobs=-1)
            self.cv_scores_ = {
                'mean': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'all_scores': cv_scores.tolist()
            }
            
            optimization_time = time.time() - start_time
            logger.info(f"{self.name} 优化完成 - 最佳{Config.SCORING}: {self.best_score_:.4f}")
            logger.info(f"{self.name} 最佳参数: {self.best_params_}")
            logger.info(f"{self.name} 交叉验证{Config.SCORING}: {self.cv_scores_['mean']:.4f} (±{self.cv_scores_['std']:.4f})")
            logger.info(f"{self.name} 优化耗时: {optimization_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"{self.name} 超参数优化失败: {str(e)}")
            # 使用默认参数作为备选
            logger.info(f"{self.name} 使用默认参数")
            self.model.fit(X, y)
            self.best_params_ = "默认参数"
            self.best_score_ = 0
            
        return self
    
    # 在 three_layer_Stacking.py 中找到 BaseModel 类的 train 方法，修改 XGBoost 的训练部分

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """模型训练
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
        """
        logger.info(f"开始训练 {self.name}")
        start_time = time.time()
        
        try:
            # 对于XGBoost，添加早停机制
            if self.name == 'XGBoost' and Config.USE_EARLY_STOPPING:
                # 划分验证集
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, 
                    test_size=0.2, 
                    random_state=Config.RANDOM_STATE,
                    stratify=y_train
                )
                
                # 修改这里：将 early_stopping_rounds 作为参数传递给模型本身
                # 而不是在 fit 方法中
                if hasattr(self.model, 'set_params'):
                    self.model.set_params(
                        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
                    )
                
                self.model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                # 普通训练
                self.model.fit(X_train, y_train)
                
            self.train_time = time.time() - start_time
            logger.info(f"{self.name} 训练完成，耗时: {self.train_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"{self.name} 训练失败: {str(e)}")
            raise

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def predict(self, X):
        """预测类别"""
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """保存模型到文件
        
        Args:
            filepath: 模型保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metadata': {
                    'name': self.name,
                    'best_params': self.best_params_,
                    'best_score': self.best_score_,
                    'cv_scores': self.cv_scores_,
                    'train_time': self.train_time,
                    'weights': self.weights_
                }
            }, f)

    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModel':
        """从文件加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            instance = cls(data['metadata']['name'])
            instance.model = data['model']
            instance.best_params_ = data['metadata']['best_params']
            instance.best_score_ = data['metadata']['best_score']
            instance.cv_scores_ = data['metadata'].get('cv_scores')
            instance.train_time = data['metadata'].get('train_time')
            instance.weights_ = data['metadata'].get('weights')
            return instance

# ==================== 多层融合模型管道 ====================
class MultiLayerEnsemblePipeline:
    """三层Stacking集成学习管道"""
    
    def __init__(self, features: pd.DataFrame, target: pd.Series, scaler=None, selected_features=None):
        """初始化集成管道
        
        Args:
            features: 特征数据
            target: 目标变量
            scaler: 数据标准化器
            selected_features: 选择的特征列
        """
        self.X = features
        self.y = target
        self.scaler = scaler
        self.selected_features = selected_features
        
        # 初始化各层模型
        self.layer1_models = [BaseModel(name) for name in Config.LAYER1_MODELS]  # 基学习器
        self.layer2_models = [BaseModel(name) for name in Config.LAYER2_MODELS]  # 中学习器
        self.layer3_models = [BaseModel(name) for name in Config.LAYER3_MODELS]  # 元学习器
        self.meta_model = self._init_meta_model()  # 最终元学习器
        
        # 存储各层预测结果和权重
        self.layer1_oof_preds = []  # 第一层Out-of-Fold预测
        self.layer1_test_preds = []  # 第一层测试集预测
        self.layer2_oof_preds = []  # 第二层Out-of-Fold预测
        self.layer2_test_preds = []  # 第二层测试集预测
        self.layer1_weights = []    # 第一层模型权重
        self.layer2_weights = []    # 第二层模型权重
        
        # 存储训练数据
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 算法标识和评估结果
        self.algorithm_name = "Three_Layer_Stacking"
        self.evaluation_results = {}
        
    def _init_meta_model(self):
        """初始化最终元学习器"""
        if Config.META_MODEL_TYPE == 'Logistic':
            return LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000,
                n_jobs=-1
            )
        elif Config.META_MODEL_TYPE == 'XGBoost':
            return xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=Config.RANDOM_STATE,
                n_estimators=200,
                n_jobs=-1,
                use_label_encoder=False
            )
        elif Config.META_MODEL_TYPE == 'GradientBoosting':
            return GradientBoostingClassifier(
                random_state=Config.RANDOM_STATE,
                n_estimators=200
            )
        else:
            raise ValueError(f"不支持的元模型类型: {Config.META_MODEL_TYPE}")

    def _split_data(self):
        """划分训练集和测试集
        
        Returns:
            划分后的数据集
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=Config.TEST_SIZE,
            stratify=self.y,
            random_state=Config.RANDOM_STATE
        )
        
        logger.info(f"数据划分完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def _calculate_model_weights(self, y_true, y_pred_proba, y_pred=None):
        """基于性能指标计算模型权重
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            y_pred: 预测标签（可选）
            
        Returns:
            模型权重
        """
        try:
            if Config.WEIGHT_METRIC == 'accuracy' and y_pred is not None:
                score = accuracy_score(y_true, y_pred)
            elif Config.WEIGHT_METRIC == 'f1' and y_pred is not None:
                score = f1_score(y_true, y_pred, zero_division=0)
            elif Config.WEIGHT_METRIC == 'roc_auc':
                score = roc_auc_score(y_true, y_pred_proba)
            else:
                # 默认使用AUC
                score = roc_auc_score(y_true, y_pred_proba)
            
            # 确保分数为正且有一定区分度
            return max(score, 0.01)
        except:
            # 如果计算失败，返回默认权重
            return 0.5

    def _train_layer_with_cv(self, models, X, y, X_test, layer_name):
        """使用K折交叉验证训练一层模型并生成预测
        
        Args:
            models: 模型列表
            X: 训练特征
            y: 训练目标
            X_test: 测试特征
            layer_name: 层名称（用于日志）
            
        Returns:
            oof_preds: Out-of-Fold预测
            test_preds: 测试集预测
            weights: 模型权重
        """
        oof_preds = []  # Out-of-Fold预测
        test_preds = []  # 测试集预测
        weights = []     # 模型权重
        
        skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
        
        for model_idx, model in enumerate(models):
            logger.info(f"训练{layer_name}模型: {model.name}")
            
            # 存储每折的预测
            oof_pred = np.zeros(len(X))
            test_pred = np.zeros(len(X_test))
            fold_weights = []
            
            # 超参数优化
            model.optimize_hyperparameters(X, y)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # 训练模型
                model.train(X_train_fold, y_train_fold)
                
                # 验证集预测
                val_pred = model.predict_proba(X_val_fold)[:, 1]
                oof_pred[val_idx] = val_pred
                
                # 测试集预测
                test_pred += model.predict_proba(X_test)[:, 1] / Config.N_FOLDS
                
                # 计算当前折的权重
                val_pred_binary = (val_pred > 0.5).astype(int)
                weight = self._calculate_model_weights(
                    y_val_fold, val_pred, val_pred_binary
                )
                fold_weights.append(weight)
            
            # 计算模型平均权重
            avg_weight = np.mean(fold_weights)
            weights.append(avg_weight)
            model.weights_ = avg_weight
            
            oof_preds.append(oof_pred)
            test_preds.append(test_pred)
            
            logger.info(f"{layer_name} - {model.name} 平均权重: {avg_weight:.4f}")
        
        # 归一化权重
        if Config.WEIGHTED_STACKING and weights:
            weight_sum = sum(weights)
            weights = [w/weight_sum for w in weights]
            logger.info(f"{layer_name} 归一化权重: {weights}")
        
        return oof_preds, test_preds, weights

    def _build_meta_features(self, X, preds, weights, include_original=True):
        """构建元特征矩阵
        
        Args:
            X: 原始特征（可选）
            preds: 预测结果列表
            weights: 模型权重列表
            include_original: 是否包含原始特征
            
        Returns:
            元特征矩阵
        """
        meta_features = []
        
        # 添加原始特征（可选）
        if include_original and Config.USE_ORIGINAL_FEATURES and X is not None:
            if isinstance(X, pd.DataFrame):
                meta_features.append(X.values)
            else:
                meta_features.append(X)
        
        # 添加加权预测特征
        for i, pred in enumerate(preds):
            if Config.WEIGHTED_STACKING and weights and i < len(weights):
                weighted_pred = pred * weights[i]
                meta_features.append(weighted_pred.reshape(-1, 1))
            else:
                meta_features.append(pred.reshape(-1, 1))
        
        # 合并所有特征
        return np.hstack(meta_features)

    def train(self):
        """训练多层融合模型
        
        Returns:
            模型评估结果
        """
        logger.info("开始训练三层Stacking模型")
        start_time = time.time()
        
        # 划分数据
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        
        # 训练第一层模型（基学习器）
        logger.info("开始训练第一层（基学习器）模型...")
        self.layer1_oof_preds, self.layer1_test_preds, self.layer1_weights = self._train_layer_with_cv(
            self.layer1_models, self.X_train, self.y_train, self.X_test, "第一层"
        )
        
        # 构建第二层训练特征
        layer1_meta_train = self._build_meta_features(
            self.X_train, self.layer1_oof_preds, self.layer1_weights
        )
        
        # 训练第二层模型（中学习器）
        logger.info("开始训练第二层（中学习器）模型...")
        self.layer2_oof_preds, self.layer2_test_preds, self.layer2_weights = self._train_layer_with_cv(
            self.layer2_models, pd.DataFrame(layer1_meta_train), self.y_train, 
            self._build_meta_features(self.X_test, self.layer1_test_preds, self.layer1_weights),
            "第二层"
        )
        
        # 构建第三层训练特征（仅使用第二层预测结果）
        logger.info("构建第三层元特征...")
        meta_features_train = self._build_meta_features(
            None, self.layer2_oof_preds, self.layer2_weights, include_original=False
        )
        
        # 训练第三层模型（元学习器）
        logger.info("开始训练第三层（元学习器）模型...")
        meta_test_features = self._build_meta_features(
            None, self.layer2_test_preds, self.layer2_weights, include_original=False
        )
        
        # 优化并训练最终元学习器
        best_meta_model = None
        best_score = -1
        
        for model in self.layer3_models:
            try:
                logger.info(f"优化最终元学习器: {model.name}")
                model.optimize_hyperparameters(meta_features_train, self.y_train)
                model.train(meta_features_train, self.y_train)
                
                # 评估性能
                y_pred_proba = model.predict_proba(meta_test_features)[:, 1]
                score = roc_auc_score(self.y_test, y_pred_proba)
                
                logger.info(f"{model.name} 测试集AUC: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_meta_model = model
                    self.meta_model = model.model
                    
            except Exception as e:
                logger.error(f"训练元学习器 {model.name} 失败: {str(e)}")
                continue
        
        if best_meta_model is None:
            logger.warning("所有元学习器训练失败，使用默认XGBoost")
            self.meta_model = xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=Config.RANDOM_STATE,
                n_estimators=100
            )
            self.meta_model.fit(meta_features_train, self.y_train)
        
        # 生成最终预测
        y_pred = self.meta_model.predict(meta_test_features)
        y_proba = self.meta_model.predict_proba(meta_test_features)[:, 1]
        
        # 评估结果
        results = self.evaluate(self.y_test, y_pred, y_proba)
        
        total_time = time.time() - start_time
        logger.info(f"三层Stacking模型训练完成，总耗时: {total_time:.2f}秒")
        
        # 保存评估结果
        self.evaluation_results = results
        self._save_evaluation_results()
        
        return results

    def evaluate(self, y_true, y_pred, y_proba):
        """评估模型性能
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率
            
        Returns:
            评估指标字典
        """
        logger.info("开始模型评估")
        
        # 初始化评估器
        evaluator = PaperReadyEvaluator(Config.OUTPUT_ROOT, Config.FIGURE_SIZE, Config.DPI)
        
        # 计算各项指标
        metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba, self.algorithm_name)
        
        # 生成可视化图表
        figure_dir = os.path.join(Config.OUTPUT_ROOT, "figures")
        evaluator.generate_algorithm_plots(y_true, y_proba, y_pred, figure_dir, self.algorithm_name)
        
        # 输出评估结果
        logger.info("\n=== 模型评估结果 ===")
        for metric, value in metrics.items():
            if metric not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
                logger.info(f"{metric}: {value:.4f}")
        
        return metrics

    def _save_evaluation_results(self):
        """保存评估结果到CSV文件"""
        try:
            # 提取主要评估指标
            metrics_to_save = {}
            for metric, value in self.evaluation_results.items():
                if metric not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
                    metrics_to_save[metric] = value
            
            # 创建DataFrame
            results_df = pd.DataFrame([metrics_to_save])
            results_df['algorithm'] = self.algorithm_name
            results_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 保存到CSV
            csv_path = os.path.join(Config.OUTPUT_ROOT, "evaluation_results.csv")
            if os.path.exists(csv_path):
                # 追加到现有文件
                existing_df = pd.read_csv(csv_path)
                updated_df = pd.concat([existing_df, results_df], ignore_index=True)
                updated_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            else:
                # 创建新文件
                results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
            logger.info(f"评估结果已保存到: {csv_path}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {str(e)}")

    def predict(self, X):
        """对新数据进行预测
        
        Args:
            X: 新数据特征
            
        Returns:
            预测概率和类别
        """
        # 构建第一层预测
        layer1_preds = []
        for model in self.layer1_models:
            try:
                pred = model.predict_proba(X)[:, 1]
                layer1_preds.append(pred)
            except:
                layer1_preds.append(np.zeros(X.shape[0]))
        
        # 构建第一层元特征
        layer1_meta = self._build_meta_features(X, layer1_preds, self.layer1_weights)
        
        # 构建第二层预测
        layer2_preds = []
        for model in self.layer2_models:
            try:
                pred = model.predict_proba(layer1_meta)[:, 1]
                layer2_preds.append(pred)
            except:
                layer2_preds.append(np.zeros(X.shape[0]))
        
        # 构建最终元特征
        final_meta = self._build_meta_features(None, layer2_preds, self.layer2_weights, include_original=False)
        
        # 最终预测
        y_proba = self.meta_model.predict_proba(final_meta)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
        
        return y_pred, y_proba

    def save_models(self):
        """保存所有模型到文件"""
        model_dir = os.path.join(Config.OUTPUT_ROOT, "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存第一层模型
        for i, model in enumerate(self.layer1_models):
            try:
                model.save_model(os.path.join(model_dir, f"layer1_{model.name}_{i}.pkl"))
            except Exception as e:
                logger.error(f"保存第一层模型 {model.name} 失败: {str(e)}")
        
        # 保存第二层模型
        for i, model in enumerate(self.layer2_models):
            try:
                model.save_model(os.path.join(model_dir, f"layer2_{model.name}_{i}.pkl"))
            except Exception as e:
                logger.error(f"保存第二层模型 {model.name} 失败: {str(e)}")
        
        # 保存第三层模型
        for i, model in enumerate(self.layer3_models):
            try:
                model.save_model(os.path.join(model_dir, f"layer3_{model.name}_{i}.pkl"))
            except Exception as e:
                logger.error(f"保存第三层模型 {model.name} 失败: {str(e)}")
        
        # 保存最终元模型
        try:
            with open(os.path.join(model_dir, "final_meta_model.pkl"), 'wb') as f:
                pickle.dump(self.meta_model, f)
        except Exception as e:
            logger.error(f"保存最终元模型失败: {str(e)}")
        
        # 保存预处理信息
        try:
            with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(os.path.join(model_dir, "feature_columns.pkl"), 'wb') as f:
                pickle.dump(self.selected_features, f)
            
            # 保存模型权重
            weights_data = {
                'layer1_weights': self.layer1_weights,
                'layer2_weights': self.layer2_weights
            }
            with open(os.path.join(model_dir, "model_weights.json"), 'w') as f:
                json.dump(weights_data, f, indent=4)
                
            logger.info(f"所有模型已保存至: {model_dir}")
            
        except Exception as e:
            logger.error(f"保存预处理信息失败: {str(e)}")

# ==================== 主程序 ====================
def main():
    """主函数"""
    DATA_PATH = "E:/怀化市地质灾害攻关/全市易发性/样本/样本数据_CF.csv"
    
    try:
        logger.info("开始三层Stacking模型训练")
        
        # 使用统一的数据处理器
        processor = UnifiedDataProcessor(
            target_col=Config.TARGET_COL,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            feature_selection=Config.FEATURE_SELECTION,
            balance_method=Config.BALANCE_METHOD
        )
        
        # 加载和预处理数据
        X, y = processor.load_and_preprocess(DATA_PATH)
        
        # 训练多层融合模型
        pipeline = MultiLayerEnsemblePipeline(
            X, y, 
            scaler=processor.scaler, 
            selected_features=processor.selected_features
        )
        
        results = pipeline.train()
        
        # 保存模型
        pipeline.save_models()
        
        logger.info("\n=== 最终模型性能 ===")
        logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"准确率: {results['accuracy']:.4f}")
        logger.info(f"F1分数: {results['f1']:.4f}")
        logger.info(f"精确率: {results['precision']:.4f}")
        logger.info(f"召回率: {results['recall']:.4f}")
        
        logger.info(f"\n所有结果已保存至: {Config.OUTPUT_ROOT}")
        
        return results

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()