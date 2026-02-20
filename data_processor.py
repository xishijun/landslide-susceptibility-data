# -*- coding: utf-8 -*-
"""
统一的数据处理器，确保三个算法使用相同的数据处理流程
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

class UnifiedDataProcessor:
    """统一的数据处理器，确保三个算法使用相同的数据处理流程"""
    
    def __init__(self, target_col='target', test_size=0.2, random_state=42,
                 feature_selection=True, balance_method='SMOTE',
                 exclude_columns=['lon', 'lat']):
        """
        初始化数据处理器
        
        Args:
            target_col: 目标列名
            test_size: 测试集比例
            random_state: 随机种子
            feature_selection: 是否进行特征选择
            balance_method: 类别不平衡处理方法
            exclude_columns: 需要排除的列名列表
        """
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.feature_selection = feature_selection
        self.balance_method = balance_method
        self.exclude_columns = exclude_columns
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_selector = None
        self.vif_results = None
        self.feature_importance = None
        self.tolerance_results = None
        self.X_processed = None
        
    def _handle_imbalance(self, X, y):
        """处理类别不平衡"""
        if self.balance_method == 'SMOTE':
            logger.info("使用SMOTE处理类别不平衡")
            sm = SMOTE(random_state=self.random_state)
            X, y = sm.fit_resample(X, y)
        return X, y
    
    def _calculate_vif_and_tolerance(self, X, output_dir=None):
        """
        计算方差膨胀因子(VIF)和容忍度评估多重共线性
        
        Args:
            X: 输入特征数据
            output_dir: 输出目录路径，如果为None则不保存文件
            
        Returns:
            vif_data: 包含VIF和容忍度结果的DataFrame
        """
        logger.info("计算特征的多重共线性(VIF和容忍度)")
        
        # 确保数据是数值型且没有缺失值
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        # 计算VIF和容忍度
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) 
                          for i in range(len(X_numeric.columns))]
        
        # 计算容忍度 (Tolerance = 1/VIF)
        vif_data["Tolerance"] = 1 / vif_data["VIF"]
        
        # 排序
        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        self.vif_results = vif_data
        self.tolerance_results = vif_data[["Feature", "Tolerance"]].sort_values(by="Tolerance", ascending=True)
        
        # 保存VIF和容忍度结果到CSV文件
        if output_dir:
            try:
                # 确保输出目录存在
                os.makedirs(output_dir, exist_ok=True)
                
                # 保存VIF结果
                vif_file_path = os.path.join(output_dir, "vif_results.csv")
                vif_data.to_csv(vif_file_path, index=False, encoding='utf-8-sig')
                logger.info(f"VIF结果已保存至: {vif_file_path}")
                
                # 保存容忍度结果
                tolerance_file_path = os.path.join(output_dir, "tolerance_results.csv")
                self.tolerance_results.to_csv(tolerance_file_path, index=False, encoding='utf-8-sig')
                logger.info(f"容忍度结果已保存至: {tolerance_file_path}")
                
            except Exception as e:
                logger.error(f"保存VIF和容忍度结果失败: {str(e)}")
        
        logger.info("多重共线性分析完成")
        return vif_data
    
    def _evaluate_feature_importance(self, X, y):
        """
        评估特征重要性
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            feature_importance_df: 特征重要性DataFrame
        """
        logger.info("评估特征重要性")
        
        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        
        # 获取特征重要性
        importance = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        logger.info("特征重要性评估完成")
        return feature_importance_df
    
    def _feature_selection(self, X, y):
        """
        特征选择方法
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            选择后的特征数据
        """
        if self.feature_selection:
            logger.info("执行特征选择")
            # 方法1: 基于随机森林的特征重要性
            rf_selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                threshold='median'
            )
            rf_selector.fit(X, y)
            
            # 方法2: 基于ANOVA F值的特征选择
            k = min(20, X.shape[1])  # 选择前k个特征
            fvalue_selector = SelectKBest(f_classif, k=k)
            fvalue_selector.fit(X, y)
            
            # 合并两种方法的选择结果
            selected_mask = rf_selector.get_support() | fvalue_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            self.feature_selector = rf_selector
            
            logger.info(f"从 {X.shape[1]} 个特征中选择了 {len(self.selected_features)} 个特征")
            return X.loc[:, self.selected_features]
        return X
    
    def generate_diagnostic_plots(self, output_path=None):
        """
        生成诊断图表
        
        Args:
            output_path: 输出目录路径
        """
        if output_path is None:
            output_path = "./data_diagnostics"
        
        os.makedirs(output_path, exist_ok=True)
        
        # 设置Catena期刊风格的图表
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        
        # 1. 绘制VIF图
        if self.vif_results is not None:
            plt.figure(figsize=(12, 8))
            
            # 创建子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # VIF图
            colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'steelblue' 
                     for vif in self.vif_results['VIF']]
            
            bars = ax1.barh(range(len(self.vif_results)), self.vif_results['VIF'], color=colors)
            ax1.set_yticks(range(len(self.vif_results)))
            ax1.set_yticklabels(self.vif_results['Feature'])
            ax1.set_xlabel('Variance Inflation Factor (VIF)', fontweight='bold')
            ax1.set_title('Multicollinearity Analysis (VIF)', fontweight='bold')
            
            # 添加阈值线
            ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='VIF > 5 (Moderate)')
            ax1.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='VIF > 10 (High)')
            ax1.legend()
            
            # 容忍度图
            colors_tol = ['red' if tol < 0.1 else 'orange' if tol < 0.2 else 'steelblue' 
                         for tol in self.tolerance_results['Tolerance']]
            
            bars = ax2.barh(range(len(self.tolerance_results)), self.tolerance_results['Tolerance'], color=colors_tol)
            ax2.set_yticks(range(len(self.tolerance_results)))
            ax2.set_yticklabels(self.tolerance_results['Feature'])
            ax2.set_xlabel('Tolerance (1/VIF)', fontweight='bold')
            ax2.set_title('Tolerance Analysis', fontweight='bold')
            
            # 添加阈值线
            ax2.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, label='Tolerance < 0.1 (High Collinearity)')
            ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Tolerance < 0.2 (Moderate Collinearity)')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "multicollinearity_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 绘制特征重要性图
        if self.feature_importance is not None:
            plt.figure(figsize=(12, 8))
            top_features = self.feature_importance.head(15)  # 显示前15个最重要特征
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Feature Importance', fontweight='bold')
            plt.title('Feature Importance Ranking', fontweight='bold')
            
            plt.gca().invert_yaxis()  # 最重要的特征显示在顶部
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 绘制相关性热力图
        if self.X_processed is not None:
            plt.figure(figsize=(12, 10))
            
            # 计算相关性矩阵
            corr_matrix = self.X_processed.corr()
            
            # 创建热力图
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0, 
                       square=True, annot=True, fmt='.2f',
                       cbar_kws={"shrink": .8})
            
            plt.title('Feature Correlation Heatmap', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"诊断图表已保存至: {output_path}")
    
    def _exclude_columns(self, X):
        """
        排除指定的列（如经纬度）
        
        Args:
            X: 输入数据
            
        Returns:
            排除指定列后的数据
        """
        if self.exclude_columns:
            columns_to_exclude = [col for col in self.exclude_columns if col in X.columns]
            if columns_to_exclude:
                logger.info(f"排除列: {columns_to_exclude}")
                X = X.drop(columns=columns_to_exclude)
        return X
    
    def load_and_preprocess(self, data_path, output_dir=None):
        """
        加载和预处理数据
        
        Args:
            data_path: 数据文件路径
            output_dir: 输出目录路径，用于保存VIF和容忍度结果
            
        Returns:
            X: 处理后的特征数据
            y: 目标变量
        """
        try:
            logger.info(f"加载数据: {data_path}")
            
            # 支持多种格式
            if data_path.endswith('.csv'):
                raw_data = pd.read_csv(data_path, encoding='gbk')
            elif data_path.endswith('.xlsx'):
                raw_data = pd.read_excel(data_path)
            else:
                raise ValueError("不支持的文件格式")
            
            if self.target_col not in raw_data.columns:
                raise ValueError(f"目标列 '{self.target_col}' 不存在")

            # 分离特征和目标
            X = raw_data.drop(columns=[self.target_col])
            y = raw_data[self.target_col]

            # 排除指定列（如经纬度）
            X = self._exclude_columns(X)

            # 处理缺失值
            if X.isnull().any().any():
                logger.info("处理缺失值")
                X = X.fillna(X.median())

            # 多重共线性分析
            self._calculate_vif_and_tolerance(X, output_dir)
            
            # 特征重要性评估
            self._evaluate_feature_importance(X, y)

            # 特征选择
            X = self._feature_selection(X, y)

            # 保存处理后的数据用于后续分析
            self.X_processed = X.copy()

            # 数据标准化
            logger.info("数据标准化")
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

            # 处理类别不平衡
            X, y = self._handle_imbalance(X, y)

            logger.info(f"数据预处理完成，最终特征维度: {X.shape}")
            return X, y

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise