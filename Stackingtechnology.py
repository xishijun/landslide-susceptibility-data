# -*- coding: utf-8 -*-
"""
融合型滑坡预测模型代码
"""
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from sklearn.base import BaseEstimator
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import seaborn as sns
import json
import logging

warnings.filterwarnings('ignore')
plt.style.use('seaborn')
sns.set(style='whitegrid', palette='muted')

# ==================== 全局配置 ====================
class Config:
    # 数据配置
    TARGET_COL = 'target'
    TEST_SIZE = 0.3
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 模型配置
    BASE_MODELS = [
        'Logistic', 'SVM', 'RandomForest', 'AdaBoost', 'XGBoost'
    ]
    META_MODEL = 'XGBoost'  # 可选 Logistic/XGBoost
    
    # 优化配置
    N_ITER = 30
    CV_FOLDS = 5
    SCORING = 'roc_auc'
    
    # 特征配置
    FEATURE_SELECTION = True
    BALANCE_METHOD = 'SMOTE'  # None/SMOTE
    
    # 路径配置
    OUTPUT_ROOT = "C:/python_deepstudy/landslide/result/ratio_1_1/1km"
    FIGURE_SIZE = (10, 6)
# ==================== 全局配置 ====================
class Config:
    # 数据配置
    TARGET_COL = 'target'
    TEST_SIZE = 0.3
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
    
    # 模型配置
    BASE_MODELS = [
        'Logistic', 'SVM', 'RandomForest', 'AdaBoost', 'XGBoost'
    ]
    META_MODEL = 'XGBoost'  # 可选 Logistic/XGBoost
    
    # 优化配置
    N_ITER = 30
    CV_FOLDS = 5
    SCORING = 'roc_auc'
    
    # 特征配置
    FEATURE_SELECTION = True
    BALANCE_METHOD = 'SMOTE'  # None/SMOTE
    
    # 路径配置
    OUTPUT_ROOT = "C:/python_deepstudy/landslide/result/ratio_1_1/1km"
    FIGURE_SIZE = (10, 6)

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("landslide_model.log"),
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
                'random_state': Config.RANDOM_STATE
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
                'random_state': Config.RANDOM_STATE
            },
            'param_space': {
                'n_estimators': Integer(200, 800),
                'max_depth': Integer(5, 30),
                'min_samples_split': Integer(2, 15)
            }
        },
        'AdaBoost': {
            'class': AdaBoostClassifier,
            'params': {
                'random_state': Config.RANDOM_STATE
            },
            'param_space': {
                'n_estimators': Integer(100, 600),
                'learning_rate': Real(0.01, 2.0),
                'algorithm': Categorical(['SAMME.R'])
            }
        },
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': 'binary:logistic',
                'n_jobs': -1,
                'random_state': Config.RANDOM_STATE
            },
            'param_space': {
                'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 12),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0, 2),
                'n_estimators': Integer(200, 1000)
            }
        }
    }

    def __init__(self, name: str):
        if name not in self.model_config:
            raise ValueError(f"不支持的模型类型: {name}")
            
        self.name = name
        cfg = self.model_config[name]
        self.model = cfg['class'](**cfg['params'])
        self.param_space = cfg['param_space']
        self.best_params_ = None
        self.best_score_ = None

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """执行贝叶斯超参数优化"""
        opt = BayesSearchCV(
            estimator=self.model,
            search_spaces=self.param_space,
            n_iter=Config.N_ITER,
            cv=StratifiedKFold(Config.CV_FOLDS),
            scoring=Config.SCORING,
            n_jobs=-1,
            random_state=Config.RANDOM_STATE
        )

        with tqdm(total=Config.N_ITER, desc=f"{self.name} 超参数优化") as pbar:
            best_score = -np.inf
            def callback(res):
                nonlocal best_score
                current_score = res.best_score_
                if current_score > best_score:
                    best_score = current_score
                pbar.set_postfix(best_score=f"{best_score:.4f}")
                pbar.update(1)

            opt.fit(X, y, callback=callback)

        self.model = opt.best_estimator_
        self.best_params_ = opt.best_params_
        self.best_score_ = opt.best_score_
        logger.info(f"{self.name} 最佳参数: {self.best_params_}")
        logger.info(f"{self.name} 最佳得分: {self.best_score_:.4f}")
        return self

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """带进度条的模型训练"""
        if hasattr(self.model, 'n_estimators'):
            total_iters = getattr(self.model, 'n_estimators', 100)
            with tqdm(total=total_iters, desc=f"{self.name} 训练") as pbar:
                if isinstance(self.model, xgb.XGBClassifier):
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train)],
                        verbose=0,
                        callbacks=[self._get_xgb_callback(pbar)]
                    )
                else:
                    self.model.fit(X_train, y_train)
                    pbar.update(total_iters)
        else:
            with tqdm(total=1, desc=f"{self.name} 训练") as pbar:
                self.model.fit(X_train, y_train)
                pbar.update(1)

    def _get_xgb_callback(self, pbar):
        def callback(env):
            pbar.update(1)
            if env.evaluation_result_list:
                last_eval = env.evaluation_result_list[-1]
                if len(last_eval) >= 2:
                    pbar.set_postfix(eval_metric=f"{last_eval[1]:.4f}")
        return callback

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metadata': {
                    'name': self.name,
                    'best_params': self.best_params_
                }
            }, f)

    @classmethod
    def load_model(cls, filepath: str) -> 'BaseModel':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            instance = cls(data['metadata']['name'])
            instance.model = data['model']
            instance.best_params_ = data['metadata']['best_params']
            return instance

# ==================== 数据预处理 ====================
class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = None
        self.selected_features = None

    def _handle_imbalance(self, X, y):
        if Config.BALANCE_METHOD == 'SMOTE':
            sm = SMOTE(random_state=Config.RANDOM_STATE)
            X, y = sm.fit_resample(X, y)
        return X, y

    def _feature_selection(self, X, y):
        if Config.FEATURE_SELECTION:
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE))
            selector.fit(X, y)
            self.selected_features = X.columns[selector.get_support()]
            return X.loc[:, self.selected_features]
        return X

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            raw_data = pd.read_csv(self.data_path, encoding='gbk')
            
            if Config.TARGET_COL not in raw_data.columns:
                raise ValueError(f"目标列{Config.TARGET_COL}不存在")

            # 分离特征和目标
            X = raw_data.drop(columns=[Config.TARGET_COL])
            y = raw_data[Config.TARGET_COL]

            # 处理缺失值
            if X.isnull().any().any():
                X = X.fillna(X.median())

            # 特征选择
            X = self._feature_selection(X, y)

            # 数据标准化
            self.scaler = StandardScaler()
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

            # 处理类别不平衡
            X, y = self._handle_imbalance(X, y)

            logger.info(f"数据预处理完成，最终特征维度: {X.shape}")
            return X, y, self.scaler

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

# ==================== 融合模型管道 ====================
class EnsemblePipeline:
    def __init__(self, features: pd.DataFrame, target: pd.Series):
        self.X = features
        self.y = target
        self.base_models = [BaseModel(name) for name in Config.BASE_MODELS]
        self.meta_model = self._init_meta_model()
        self.meta_features = None

    def _init_meta_model(self):
        if Config.META_MODEL == 'Logistic':
            return LogisticRegression()
        elif Config.META_MODEL == 'XGBoost':
            return xgb.XGBClassifier(objective='binary:logistic')
        else:
            raise ValueError(f"不支持的元模型类型: {Config.META_MODEL}")

    def _split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=Config.TEST_SIZE,
            stratify=self.y,
            random_state=Config.RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=Config.VAL_SIZE,
            stratify=y_train,
            random_state=Config.RANDOM_STATE
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _train_base_models(self, X_train, y_train):
        for model in self.base_models:
            try:
                model.optimize_hyperparameters(X_train, y_train)
                model.train(X_train, y_train)
            except Exception as e:
                logger.error(f"{model.name} 训练失败: {str(e)}")
                raise

    def _build_meta_features(self, X):
        meta_features = []
        for model in self.base_models:
            proba = model.predict_proba(X)[:, 1]
            meta_features.append(proba.reshape(-1, 1))
        return np.hstack([X, np.hstack(meta_features)])

    def train(self):
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data()
        
        # 训练基模型
        self._train_base_models(X_train, y_train)
        
        # 构建元特征
        self.meta_features_train = self._build_meta_features(X_val)
        self.meta_features_test = self._build_meta_features(X_test)
        
        # 训练元模型
        self.meta_model.fit(self.meta_features_train, y_val)
        
        # 评估
        results = self.evaluate(self.meta_features_test, y_test)
        return results

    def evaluate(self, X_test, y_test):
        y_pred = self.meta_model.predict(X_test)
        y_proba = self.meta_model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'brier': brier_score_loss(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'class_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # 生成可视化图表
        self._generate_plots(y_test, y_proba, y_pred)
        return metrics

    def _generate_plots(self, y_true, y_proba, y_pred):
        os.makedirs(os.path.join(Config.OUTPUT_ROOT, "figures"), exist_ok=True)
        
        # ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=Config.FIGURE_SIZE)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_proba):.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(Config.OUTPUT_ROOT, "figures", "roc_curve.png"))
        plt.close()

        # PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.figure(figsize=Config.FIGURE_SIZE)
        plt.plot(recall, precision, label=f'PR AUC = {average_precision_score(y_true, y_proba):.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(Config.OUTPUT_ROOT, "figures", "pr_curve.png"))
        plt.close()

        # 特征重要性（XGBoost作为元模型时）
        if isinstance(self.meta_model, xgb.XGBClassifier):
            xgb.plot_importance(self.meta_model)
            plt.title('Feature Importance')
            plt.savefig(os.path.join(Config.OUTPUT_ROOT, "figures", "feature_importance.png"))
            plt.close()

    def save_models(self):
        model_dir = os.path.join(Config.OUTPUT_ROOT, "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存基模型
        for model in self.base_models:
            model.save_model(os.path.join(model_dir, f"{model.name}_base.pkl"))
        
        # 保存元模型
        with open(os.path.join(model_dir, "meta_model.pkl"), 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        # 保存预处理信息
        with open(os.path.join(model_dir, "preprocessor.pkl"), 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'selected_features': self.selected_features
            }, f)

# ==================== 主程序 ====================
if __name__ == "__main__":
    DATA_PATH = "C:/python_deepstudy/landslide/result/ratio_1_1/平衡后的正负样本.csv"
    
    try:
        # 数据预处理
        processor = DataProcessor(DATA_PATH)
        X, y, scaler = processor.load_and_preprocess()
        
        # 训练融合模型
        pipeline = EnsemblePipeline(X, y)
        results = pipeline.train()
        
        # 保存结果
        pipeline.save_models()
        logger.info("\n模型评估结果:\n" + json.dumps(results, indent=4))
        
        # 保存评估报告
        report_path = os.path.join(Config.OUTPUT_ROOT, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"模型和评估报告已保存至: {Config.OUTPUT_ROOT}")

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise