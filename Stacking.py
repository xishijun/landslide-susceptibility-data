# -*- coding: utf-8 -*-
"""
滑坡预测模型 - 贝叶斯优化+Stacking集成（传统算法）
论文出版优化版本
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    StackingClassifier
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
    brier_score_loss
)
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import seaborn as sns
import json
import logging
import time
import sklearn

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
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # 模型配置
    BASE_MODELS = [
        'Logistic', 'SVM', 'RandomForest', 'AdaBoost', 'XGBoost'
    ]
    FINAL_ESTIMATOR = 'Logistic'  # Stacking的最终估计器
    
    # 优化配置
    N_ITER = 3  # 贝叶斯优化迭代次数
    CV_FOLDS = 3  # 交叉验证折数
    SCORING = 'roc_auc'  # 优化目标指标
    
    # 特征配置
    FEATURE_SELECTION = True
    BALANCE_METHOD = 'SMOTE'  # None/SMOTE
    
    # 路径配置
    OUTPUT_ROOT = "E:/怀化市地质灾害攻关/全市易发性/result/传统算法"
    FIGURE_SIZE = (10, 8)
    DPI = 300  # 图表分辨率

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

# ==================== 核心模型类 ====================
class OptimizedModel:
    """使用贝叶斯优化的机器学习模型"""

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
                'penalty': Categorical(['l2', None]),
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
                'min_samples_split': Integer(2, 15),
                'min_samples_leaf': Integer(1, 10)
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
                # 根据scikit-learn版本选择合适的算法
                'algorithm': Categorical(['SAMME'])  # 只使用SAMME算法，避免兼容性问题
            }
        },
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': 'binary:logistic',
                'n_jobs': -1,
                'random_state': Config.RANDOM_STATE,
                'eval_metric': 'logloss'
            },
            'param_space': {
                'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 12),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0, 2),
                'reg_alpha': Real(0, 10),
                'reg_lambda': Real(1, 10),
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
        self.optimization_time = None

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> 'OptimizedModel':
        """执行贝叶斯超参数优化"""
        start_time = time.time()
        
        # 检查scikit-learn版本，调整AdaBoost参数空间
        if self.name == 'AdaBoost':
            sklearn_version = sklearn.__version__
            logger.info(f"scikit-learn版本: {sklearn_version}")
            
            # 根据版本调整参数空间
            if sklearn_version >= '1.2':
                # 新版本支持SAMME.R
                self.param_space['algorithm'] = Categorical(['SAMME', 'SAMME.R'])
            else:
                # 旧版本只支持SAMME
                self.param_space['algorithm'] = Categorical(['SAMME'])
        
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

        try:
            with tqdm(total=Config.N_ITER, desc=f"{self.name} 超参数优化") as pbar:
                def callback(optim_result):
                    pbar.update(1)
                    current_score = -optim_result.fun if hasattr(optim_result, 'fun') else optim_result.func_vals[-1]
                    pbar.set_postfix(best_score=f"{current_score:.4f}")

                opt.fit(X, y, callback=callback)

            self.model = opt.best_estimator_
            self.best_params_ = opt.best_params_
            self.best_score_ = opt.best_score_
            self.optimization_time = time.time() - start_time
            
            logger.info(f"{self.name} 最佳参数: {self.best_params_}")
            logger.info(f"{self.name} 最佳{Config.SCORING}: {self.best_score_:.4f}")
            logger.info(f"{self.name} 优化耗时: {self.optimization_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"{self.name} 超参数优化失败: {str(e)}")
            # 使用默认参数作为备选方案
            logger.info(f"{self.name} 使用默认参数")
            self.model.fit(X, y)
            self.best_params_ = "使用默认参数"
            self.best_score_ = 0
            self.optimization_time = time.time() - start_time
        
        return self
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """模型训练"""
        with tqdm(total=1, desc=f"{self.name} 训练") as pbar:
            self.model.fit(X_train, y_train)
            pbar.update(1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    
    def cross_val_score(self, X, y, cv=5):
        """交叉验证评估"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=Config.SCORING)
        return scores.mean(), scores.std()

    def save_model(self, filepath: str) -> None:
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metadata': {
                    'name': self.name,
                    'best_params': self.best_params_,
                    'best_score': self.best_score_,
                    'optimization_time': self.optimization_time
                }
            }, f)

    @classmethod
    def load_model(cls, filepath: str) -> 'OptimizedModel':
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            instance = cls(data['metadata']['name'])
            instance.model = data['model']
            instance.best_params_ = data['metadata']['best_params']
            instance.best_score_ = data['metadata']['best_score']
            instance.optimization_time = data['metadata']['optimization_time']
            return instance

# ==================== Stacking集成模型 ====================
class StackingEnsemble:
    def __init__(self, features: pd.DataFrame, target: pd.Series):
        self.X = features
        self.y = target
        self.base_models = [OptimizedModel(name) for name in Config.BASE_MODELS]
        self.stacking_model = None
        self.cv_scores = {}
        self.algorithm_name = "Traditional_Stacking"  # 标识算法类型
        
    def _init_final_estimator(self):
        """初始化最终估计器"""
        if Config.FINAL_ESTIMATOR == 'Logistic':
            return LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000)
        elif Config.FINAL_ESTIMATOR == 'XGBoost':
            return xgb.XGBClassifier(
                objective='binary:logistic', 
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
        else:
            raise ValueError(f"不支持的最终估计器类型: {Config.FINAL_ESTIMATOR}")

    def _split_data(self):
        """划分训练测试集"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=Config.TEST_SIZE,
            stratify=self.y,
            random_state=Config.RANDOM_STATE
        )
        return X_train, X_test, y_train, y_test

    def _optimize_base_models(self, X_train, y_train):
        """优化所有基模型的超参数"""
        logger.info("开始优化基模型超参数")
        for model in self.base_models:
            try:
                model.optimize_hyperparameters(X_train, y_train)
                # 交叉验证评估
                mean_score, std_score = model.cross_val_score(X_train, y_train, cv=Config.CV_FOLDS)
                self.cv_scores[model.name] = {'mean': mean_score, 'std': std_score}
                logger.info(f"{model.name} 交叉验证{Config.SCORING}: {mean_score:.4f} (±{std_score:.4f})")
            except Exception as e:
                logger.error(f"{model.name} 超参数优化失败: {str(e)}")
                # 即使优化失败，也尝试使用默认参数进行交叉验证
                try:
                    mean_score, std_score = model.cross_val_score(X_train, y_train, cv=Config.CV_FOLDS)
                    self.cv_scores[model.name] = {'mean': mean_score, 'std': std_score}
                    logger.info(f"{model.name} 使用默认参数交叉验证{Config.SCORING}: {mean_score:.4f} (±{std_score:.4f})")
                except Exception as inner_e:
                    logger.error(f"{model.name} 交叉验证也失败: {str(inner_e)}")
                    # 如果交叉验证也失败，记录一个默认的低分
                    self.cv_scores[model.name] = {'mean': 0.5, 'std': 0.1}

    def train(self):
        """训练Stacking集成模型"""
        X_train, X_test, y_train, y_test = self._split_data()
        
        # 优化基模型超参数
        self._optimize_base_models(X_train, y_train)
        
        # 准备Stacking的基估计器列表
        estimators = [(model.name, model.model) for model in self.base_models]
        
        # 创建Stacking分类器
        final_estimator = self._init_final_estimator()
        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=Config.CV_FOLDS,
            n_jobs=-1,
            verbose=1
        )
        
        # 训练Stacking模型
        logger.info("开始训练Stacking模型")
        start_time = time.time()
        self.stacking_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        logger.info(f"Stacking模型训练完成，耗时: {training_time:.2f}秒")
        
        # 评估
        results = self.evaluate(X_test, y_test)
        return results

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        logger.info("开始模型评估")
        
        # Stacking模型预测
        y_pred = self.stacking_model.predict(X_test)
        y_proba = self.stacking_model.predict_proba(X_test)[:, 1]

        # 初始化评估器
        evaluator = PaperReadyEvaluator(Config.OUTPUT_ROOT, Config.FIGURE_SIZE, Config.DPI)
        
        # 计算各项指标
        metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba, self.algorithm_name)
        
        # 生成可视化图表
        self._generate_plots(y_test, y_proba, y_pred, evaluator)
        
        # 记录评估结果
        logger.info("\n模型评估结果:")
        for metric, value in metrics.items():
            if metric not in ['class_report', 'confusion_matrix', 'roc_curve', 'pr_curve']:
                logger.info(f"{metric}: {value:.4f}")
        
        return metrics

    def _generate_plots(self, y_true, y_proba, y_pred, evaluator):
        """生成评估图表"""
        figures_dir = os.path.join(Config.OUTPUT_ROOT, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # 使用评估器生成高质量图表
        evaluator.generate_algorithm_plots(y_true, y_proba, y_pred, figures_dir, self.algorithm_name)
        
        # 基模型性能比较
        if hasattr(self, 'cv_scores') and self.cv_scores:
            valid_scores = {}
            for model_name, score_data in self.cv_scores.items():
                if isinstance(score_data, dict) and 'mean' in score_data and 'std' in score_data:
                    if not np.isnan(score_data['mean']) and not np.isnan(score_data['std']):
                        valid_scores[model_name] = score_data
            
            if valid_scores:
                model_names = list(valid_scores.keys())
                means = [valid_scores[name]['mean'] for name in model_names]
                stds = [valid_scores[name]['std'] for name in model_names]
                
                # 创建中文名称映射
                chinese_names = {
                    'Logistic': 'Logistic Regression',
                    'SVM': 'SVM',
                    'RandomForest': 'Random Forest',
                    'AdaBoost': 'AdaBoost',
                    'XGBoost': 'XGBoost'
                }
                display_names = [chinese_names.get(name, name) for name in model_names]
                
                plt.figure(figsize=Config.FIGURE_SIZE)
                y_pos = np.arange(len(model_names))
                bars = plt.barh(y_pos, means, xerr=stds, align='center', alpha=0.8)
                plt.yticks(y_pos, display_names)
                plt.xlabel(f'{Config.SCORING} Score')
                plt.title('Base Models Performance Comparison')
                plt.xlim(0, 1)
                
                # 添加数值标签
                for i, (mean, std) in enumerate(zip(means, stds)):
                    plt.text(mean + 0.01, i, f'{mean:.3f} ± {std:.3f}', va='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(figures_dir, "base_models_comparison.png"), 
                           dpi=Config.DPI, bbox_inches='tight')
                plt.close()
                
                # 保存基模型性能数据
                performance_df = pd.DataFrame({
                    'Model': display_names,
                    'Mean_Score': means,
                    'Std_Dev': stds
                })
                performance_df.to_csv(os.path.join(Config.OUTPUT_ROOT, "base_models_performance.csv"), 
                                    index=False, encoding='utf-8-sig')

    def save_models(self):
        """保存所有模型"""
        model_dir = os.path.join(Config.OUTPUT_ROOT, "saved_models")
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存基模型
        for model in self.base_models:
            model.save_model(os.path.join(model_dir, f"{model.name}_base.pkl"))
        
        # 保存Stacking模型
        with open(os.path.join(model_dir, "stacking_model.pkl"), 'wb') as f:
            pickle.dump(self.stacking_model, f)
        
        logger.info(f"所有模型已保存至: {model_dir}")

# ==================== 结果报告生成 ====================
class ReportGenerator:
    @staticmethod
    def generate_report(ensemble, results, processor, data_path):
        """生成完整的结果报告"""
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'data_info': {
                'source': data_path,
                'original_features': len(processor.selected_features) if processor.selected_features else "未进行特征选择",
                'target_distribution': {
                    'class_0': sum(ensemble.y == 0),
                    'class_1': sum(ensemble.y == 1)
                }
            },
            'config': {
                'base_models': Config.BASE_MODELS,
                'final_estimator': Config.FINAL_ESTIMATOR,
                'test_size': Config.TEST_SIZE,
                'random_state': Config.RANDOM_STATE,
                'feature_selection': Config.FEATURE_SELECTION,
                'balance_method': Config.BALANCE_METHOD
            },
            'base_models_performance': ensemble.cv_scores,
            'stacking_performance': {k: v for k, v in results.items() if k not in ['class_report', 'confusion_matrix', 'roc_curve', 'pr_curve']},
            'optimization_times': {model.name: model.optimization_time for model in ensemble.base_models}
        }
        
        # 保存报告
        report_path = os.path.join(Config.OUTPUT_ROOT, "evaluation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        
        logger.info(f"评估报告已保存至: {report_path}")
        return report

# ==================== 主程序 ====================
def main():
    # 数据路径
    DATA_PATH = "E:/怀化市地质灾害攻关/全市易发性/样本/样本数据_CF.csv"
    
    try:
        # 使用统一的数据处理器
        processor = UnifiedDataProcessor(
            target_col=Config.TARGET_COL,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            feature_selection=Config.FEATURE_SELECTION,
            balance_method=Config.BALANCE_METHOD
        )
        X, y = processor.load_and_preprocess(DATA_PATH)
        
        # 训练Stacking集成模型
        ensemble = StackingEnsemble(X, y)
        results = ensemble.train()
        
        # 保存模型
        ensemble.save_models()
        
        # 生成报告
        report = ReportGenerator.generate_report(ensemble, results, processor, DATA_PATH)
        
        logger.info("\n=== 最终模型性能 ===")
        logger.info(f"ROC-AUC: {results['roc_auc']:.4f}")
        logger.info(f"准确率: {results['accuracy']:.4f}")
        logger.info(f"F1分数: {results['f1']:.4f}")
        logger.info(f"精确率: {results['precision']:.4f}")
        logger.info(f"召回率: {results['recall']:.4f}")
        
        logger.info(f"\n所有结果已保存至: {Config.OUTPUT_ROOT}")

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()