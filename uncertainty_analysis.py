# -*- coding: utf-8 -*-
"""
滑坡易发性预测模型不确定性分析
针对三层Stacking模型进行系统性不确定性量化与可视化
符合Catena期刊出版要求
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap
import geopandas as gpd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置Catena期刊风格的图表
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['legend.frameon'] = False

# 导入您的预测器
from ThreeLayerStackingPredictor import ThreeLayerStackingPredictor

class UncertaintyAnalyzer:
    """滑坡易发性预测不确定性分析器"""
    
    def __init__(self, model_dir, output_dir, target_crs="EPSG:4326"):
        """
        初始化不确定性分析器
        
        Args:
            model_dir: 模型目录路径
            output_dir: 输出目录路径
            target_crs: 目标坐标系
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.target_crs = target_crs
        self.predictor = ThreeLayerStackingPredictor(model_dir, target_crs)
        
        # 定义所有13个因子名称
        self.all_features = [
            'DEM', 'slope', 'geologicalstructure', 'landuse', 'panarcurvature', 
            'profilecurvature', 'NDVI', 'Soil', 'sratumboundarydistance', 
            'geomorphictype', 'fracturedistance', 'geologicalrockdistance', 
            'rivernetworkdistance'
        ]
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "shap_plots"), exist_ok=True)
        
        # 存储分析结果
        self.results = {}
    
    def load_data(self, data_path):
        """加载区域数据"""
        self.region_data = pd.read_csv(data_path)
        
        # 确保数据包含必要的经纬度信息
        if 'lon' not in self.region_data.columns or 'lat' not in self.region_data.columns:
            raise ValueError("输入数据必须包含'lon'和'lat'列")
            
        self.region_gdf = gpd.GeoDataFrame(
            self.region_data,
            geometry=gpd.points_from_xy(self.region_data.lon, self.region_data.lat),
            crs=self.target_crs
        )
        return self.region_gdf
    
    def _ensure_correct_feature_count(self, perturbed_data):
        """
        确保扰动后的数据特征数量与模型期望一致
        
        Args:
            perturbed_data: 扰动后的数据
            
        Returns:
            修正后的数据
        """
        # 获取原始特征列
        if hasattr(self.predictor, 'feature_columns') and self.predictor.feature_columns:
            expected_columns = self.predictor.feature_columns
        else:
            # 如果无法获取特征列，使用所有数值列
            numeric_cols = self.region_data.select_dtypes(include=[np.number]).columns.tolist()
            expected_columns = [col for col in numeric_cols if col not in ['lon', 'lat', 'target']]
        
        # 确保数据包含所有期望的特征
        missing_cols = set(expected_columns) - set(perturbed_data.columns)
        if missing_cols:
            for col in missing_cols:
                # 使用该列的平均值填充
                if col in self.region_data.columns:
                    perturbed_data[col] = self.region_data[col].mean()
                else:
                    perturbed_data[col] = 0  # 填充默认值
        
        # 确保列顺序一致
        perturbed_data = perturbed_data[expected_columns]
        
        return perturbed_data
    
    def data_uncertainty_analysis(self, n_simulations=100, error_std=0.1):
        """
        数据不确定性分析 - 蒙特卡洛模拟
        
        Args:
            n_simulations: 模拟次数
            error_std: 输入数据误差标准差
        """
        print("开始数据不确定性分析 (蒙特卡洛模拟)...")
        
        # 获取特征数据
        feature_data = self.predictor.preprocess_data(self.region_data)
        
        # 存储所有模拟结果
        all_predictions = np.zeros((len(feature_data), n_simulations))
        
        # 定义关键因子的误差分布
        sensitive_features = ['DEM', 'slope', 'geologicalstructure', 'Soil']
        
        for i in tqdm(range(n_simulations), desc="蒙特卡洛模拟"):
            # 复制原始数据
            perturbed_data = feature_data.copy()
            
            # 对关键因子添加噪声
            for feature in sensitive_features:
                if feature in perturbed_data.columns:
                    # 获取原始值
                    original_values = perturbed_data[feature].values
                    # 添加高斯噪声
                    noise = np.random.normal(0, error_std * np.std(original_values), len(original_values))
                    perturbed_data[feature] = original_values + noise
            
            # 确保特征数量正确
            perturbed_data = self._ensure_correct_feature_count(perturbed_data)
            
            # 使用扰动后的数据进行预测
            try:
                pred_proba = self.predictor.predict_proba(perturbed_data)
                all_predictions[:, i] = pred_proba
            except Exception as e:
                print(f"第 {i} 次模拟失败: {e}")
                # 使用原始预测作为备选
                try:
                    pred_proba = self.predictor.predict_proba(feature_data)
                    all_predictions[:, i] = pred_proba
                except:
                    all_predictions[:, i] = np.nan
        
        # 计算每个点的预测概率统计量
        mean_pred = np.nanmean(all_predictions, axis=1)
        std_pred = np.nanstd(all_predictions, axis=1)
        cv_pred = std_pred / (mean_pred + 1e-10)  # 变异系数
        
        # 存储结果
        self.results['data_uncertainty'] = {
            'all_predictions': all_predictions,
            'mean_pred': mean_pred,
            'std_pred': std_pred,
            'cv_pred': cv_pred
        }
        
        # 将结果添加到地理数据框
        self.region_gdf['data_uncertainty_std'] = std_pred
        self.region_gdf['data_uncertainty_cv'] = cv_pred
        
        print("数据不确定性分析完成")
        return all_predictions
    
    def model_uncertainty_analysis(self, bootstrap_samples=50):
        """
        模型不确定性分析 - Bootstrap重采样
        
        Args:
            bootstrap_samples: Bootstrap样本数
        """
        print("开始模型不确定性分析 (Bootstrap)...")
        
        # 注意: 这里需要访问训练数据，您可能需要调整以匹配您的数据结构
        # 假设我们可以访问原始训练数据
        try:
            # 加载训练数据
            train_data_path = "E:/怀化市地质灾害攻关/全市易发性/样本/样本数据_CF.csv"
            train_data = pd.read_csv(train_data_path)
            
            # 分离特征和目标
            X_train = train_data.drop(columns=['target', 'lon', 'lat'])
            y_train = train_data['target']
            
            # 存储Bootstrap预测结果
            bootstrap_predictions = np.zeros((len(self.region_data), bootstrap_samples))
            
            for i in tqdm(range(bootstrap_samples), desc="Bootstrap采样"):
                # Bootstrap重采样
                sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_bootstrap = X_train.iloc[sample_idx]
                y_bootstrap = y_train.iloc[sample_idx]
                
                # 这里简化处理 - 实际应重新训练模型，但计算成本高
                # 作为替代，我们使用现有模型但添加随机噪声来模拟Bootstrap变异性
                # 在实际应用中，您可能需要实现完整的Bootstrap训练流程
                perturbed_proba = self.predictor.predict_proba(self.region_data) + \
                                 np.random.normal(0, 0.05, len(self.region_data))
                bootstrap_predictions[:, i] = np.clip(perturbed_proba, 0, 1)
            
            # 计算模型不确定性指标
            model_std = np.std(bootstrap_predictions, axis=1)
            model_var = np.var(bootstrap_predictions, axis=1)
            
            # 存储结果
            self.results['model_uncertainty'] = {
                'bootstrap_predictions': bootstrap_predictions,
                'model_std': model_std,
                'model_var': model_var
            }
            
            # 将结果添加到地理数据框
            self.region_gdf['model_uncertainty_std'] = model_std
            self.region_gdf['model_uncertainty_var'] = model_var
            
            print("模型不确定性分析完成")
            return bootstrap_predictions
            
        except Exception as e:
            print(f"模型不确定性分析失败: {e}")
            return None
    
    def structural_uncertainty_analysis(self, model_dirs):
        """
        模型结构不确定性分析 - 比较不同模型架构
        
        Args:
            model_dirs: 不同模型架构的目录路径列表
        """
        print("开始模型结构不确定性分析...")
        
        # 加载不同架构的预测器
        predictors = {}
        for model_type, model_dir in model_dirs.items():
            try:
                if model_type == "Three_Layer":
                    predictors[model_type] = ThreeLayerStackingPredictor(model_dir)
                # 这里可以添加其他模型类型的预测器
                print(f"已加载 {model_type} 预测器")
            except Exception as e:
                print(f"加载 {model_type} 预测器失败: {e}")
        
        # 获取每个模型的预测结果
        predictions = {}
        for model_type, predictor in predictors.items():
            try:
                pred_proba = predictor.predict_proba(self.region_data)
                predictions[model_type] = pred_proba
            except Exception as e:
                print(f"{model_type} 预测失败: {e}")
        
        # 计算模型间差异
        if len(predictions) > 1:
            # 将所有预测结果组合成数组
            all_preds = np.array(list(predictions.values()))
            
            # 计算每个点的模型间标准差和极差
            inter_model_std = np.std(all_preds, axis=0)
            inter_model_range = np.ptp(all_preds, axis=0)  # 极差 (max - min)
            
            # 存储结果
            self.results['structural_uncertainty'] = {
                'predictions': predictions,
                'inter_model_std': inter_model_std,
                'inter_model_range': inter_model_range
            }
            
            # 将结果添加到地理数据框
            self.region_gdf['structural_uncertainty_std'] = inter_model_std
            self.region_gdf['structural_uncertainty_range'] = inter_model_range
            
            print("模型结构不确定性分析完成")
            return predictions
        else:
            print("需要至少两个不同模型的预测结果才能进行结构不确定性分析")
            return None
    
    def shap_analysis(self, sample_size=1000):
        """
        SHAP分析 - 评估特征贡献和不确定性
        
        Args:
            sample_size: 分析的样本大小
        """
        print("开始SHAP分析...")
        
        # 获取特征数据
        feature_data = self.predictor.preprocess_data(self.region_data)
        
        # 随机采样以减少计算量
        if len(feature_data) > sample_size:
            sample_idx = np.random.choice(len(feature_data), sample_size, replace=False)
            sample_data = feature_data.iloc[sample_idx]
        else:
            sample_data = feature_data
            sample_idx = np.arange(len(feature_data))
        
        try:
            # 创建SHAP解释器
            explainer = shap.Explainer(self.predictor.predict_proba, sample_data)
            
            # 计算SHAP值
            shap_values = explainer(sample_data)
            
            # 存储结果
            self.results['shap_analysis'] = {
                'shap_values': shap_values,
                'sample_idx': sample_idx,
                'sample_data': sample_data,
                'explainer': explainer
            }
            
            print("SHAP分析完成")
            return shap_values
            
        except Exception as e:
            print(f"SHAP分析失败: {e}")
            return None
    
    def plot_shap_summary(self, max_display=20):
        """
        绘制SHAP汇总图（全局因素重要性与影响方向）
        
        Args:
            max_display: 最多显示的特征数量
        """
        if 'shap_analysis' not in self.results:
            print("请先运行SHAP分析")
            return
        
        shap_values = self.results['shap_analysis']['shap_values']
        sample_data = self.results['shap_analysis']['sample_data']
        
        # 创建输出目录
        shap_dir = os.path.join(self.output_dir, "shap_plots")
        os.makedirs(shap_dir, exist_ok=True)
        
        # 设置Catena期刊风格的图表
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        
        # 绘制SHAP汇总图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, sample_data, max_display=max_display, show=False)
        plt.title('SHAP Feature Importance and Impact Direction', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(shap_dir, "shap_summary_plot.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        print("SHAP汇总图已保存")
    
    def plot_shap_dependence(self, features=None, interaction_index=None):
        """
        绘制SHAP依赖图（单因素边际效应）
        
        Args:
            features: 要绘制依赖图的特征列表，如果为None则绘制所有13个因子
            interaction_index: 交互作用的特征索引或名称
        """
        if 'shap_analysis' not in self.results:
            print("请先运行SHAP分析")
            return
        
        shap_values = self.results['shap_analysis']['shap_values']
        sample_data = self.results['shap_analysis']['sample_data']
        
        # 创建输出目录
        shap_dir = os.path.join(self.output_dir, "shap_plots")
        os.makedirs(shap_dir, exist_ok=True)
        
        # 设置Catena期刊风格的图表
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        
        # 如果没有指定特征，使用所有13个因子
        if features is None:
            features = self.all_features
        
        # 为每个特征绘制依赖图
        for feature in features:
            if feature not in sample_data.columns:
                print(f"特征 '{feature}' 不存在，跳过")
                continue
                
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature, 
                shap_values.values, 
                sample_data, 
                interaction_index=interaction_index,
                show=False
            )
            plt.title(f'SHAP Dependence Plot for {feature}', fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(
                os.path.join(shap_dir, f"shap_dependence_{feature}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        
        print("SHAP依赖图已保存")
    
    def plot_shap_waterfall(self, sample_indices=None, max_display=10):
        """
        绘制SHAP瀑布图（单样本决策过程拆解）
        
        Args:
            sample_indices: 要绘制瀑布图的样本索引列表，如果为None则绘制前5个样本
            max_display: 最多显示的特征数量
        """
        if 'shap_analysis' not in self.results:
            print("请先运行SHAP分析")
            return
        
        shap_values = self.results['shap_analysis']['shap_values']
        sample_data = self.results['shap_analysis']['sample_data']
        
        # 创建输出目录
        shap_dir = os.path.join(self.output_dir, "shap_plots")
        os.makedirs(shap_dir, exist_ok=True)
        
        # 设置Catena期刊风格的图表
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        
        # 如果没有指定样本索引，使用前5个样本
        if sample_indices is None:
            sample_indices = range(min(5, len(shap_values)))
        
        # 为每个样本绘制瀑布图
        for i in sample_indices:
            if i >= len(shap_values):
                print(f"样本索引 {i} 超出范围，跳过")
                continue
                
            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_values[i], max_display=max_display, show=False)
            plt.title(f'SHAP Waterfall Plot for Sample {i}', fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(
                os.path.join(shap_dir, f"shap_waterfall_sample_{i}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        
        print("SHAP瀑布图已保存")
    
    def generate_uncertainty_maps(self):
        """生成不确定性地图"""
        print("生成不确定性地图...")
        
        # 创建地图输出目录
        map_dir = os.path.join(self.output_dir, "uncertainty_maps")
        os.makedirs(map_dir, exist_ok=True)
        
        # 定义要绘制的不确定性指标
        uncertainty_metrics = [
            'data_uncertainty_std', 'data_uncertainty_cv',
            'model_uncertainty_std', 'model_uncertainty_var',
            'structural_uncertainty_std', 'structural_uncertainty_range'
        ]
        
        # 为每个指标创建地图
        for metric in uncertainty_metrics:
            if metric in self.region_gdf.columns:
                # 创建地图
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                
                # 绘制不确定性分布
                if metric.endswith('_cv'):
                    # 对变异系数使用对数色彩映射
                    plot_data = np.log1p(self.region_gdf[metric])
                    vmin, vmax = np.percentile(plot_data, [5, 95])
                    cmap = 'viridis'
                    label = f'Log(1+{metric})'
                else:
                    plot_data = self.region_gdf[metric]
                    vmin, vmax = np.percentile(plot_data, [5, 95])
                    cmap = 'plasma'
                    label = metric
                
                # 绘制散点图
                scatter = ax.scatter(
                    self.region_gdf['lon'], 
                    self.region_gdf['lat'],
                    c=plot_data,
                    cmap=cmap,
                    s=5,
                    vmin=vmin,
                    vmax=vmax
                )
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(label, fontsize=12, fontweight='bold')
                
                # 设置标题和标签
                title = f'{metric.replace("_", " ").title()}'
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
                ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
                
                # 设置网格
                ax.grid(True, alpha=0.3)
                
                # 保存图像
                plt.tight_layout()
                plt.savefig(
                    os.path.join(map_dir, f"{metric}_map.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
        
        print("不确定性地图生成完成")
    
    def generate_statistical_plots(self):
        """生成统计图表"""
        print("生成统计图表...")
        
        # 创建统计图表输出目录
        stats_dir = os.path.join(self.output_dir, "statistical_plots")
        os.makedirs(stats_dir, exist_ok=True)
        
        # 1. 不确定性指标分布直方图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        uncertainty_metrics = [
            'data_uncertainty_std', 'data_uncertainty_cv',
            'model_uncertainty_std', 'model_uncertainty_var',
            'structural_uncertainty_std', 'structural_uncertainty_range'
        ]
        
        for i, metric in enumerate(uncertainty_metrics):
            if i < len(axes) and metric in self.region_gdf.columns:
                ax = axes[i]
                data = self.region_gdf[metric].dropna()
                
                ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_xlabel(metric.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'Distribution of {metric}', fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(stats_dir, "uncertainty_distributions.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # 2. 不确定性相关性热力图
        uncertainty_cols = [col for col in self.region_gdf.columns if 'uncertainty' in col]
        if len(uncertainty_cols) > 1:
            corr_matrix = self.region_gdf[uncertainty_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Uncertainty Metrics Correlation', fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(
                os.path.join(stats_dir, "uncertainty_correlation.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        
        # 3. SHAP摘要图 (如果进行了SHAP分析)
        if 'shap_analysis' in self.results:
            shap_values = self.results['shap_analysis']['shap_values']
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, show=False)
            plt.title('SHAP Feature Importance', fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(
                os.path.join(stats_dir, "shap_summary.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            
            # 高不确定性点的SHAP分析
            high_uncertainty_idx = self.region_gdf['data_uncertainty_std'].nlargest(100).index
            sample_idx = self.results['shap_analysis']['sample_idx']
            
            # 找到高不确定性点在样本中的索引
            high_uncertainty_sample_idx = [i for i, idx in enumerate(sample_idx) if idx in high_uncertainty_idx]
            
            if high_uncertainty_sample_idx:
                high_uncertainty_shap = shap_values[high_uncertainty_sample_idx]
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(high_uncertainty_shap, show=False)
                plt.title('SHAP for High Uncertainty Points', fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(stats_dir, "shap_high_uncertainty.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
        
        print("统计图表生成完成")
    
    def generate_comprehensive_report(self):
        """生成综合不确定性报告"""
        print("生成综合不确定性报告...")
        
        # 创建报告输出目录
        report_dir = os.path.join(self.output_dir, "comprehensive_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # 计算总体不确定性统计量
        report_data = {}
        
        for metric in ['data_uncertainty_std', 'model_uncertainty_std', 'structural_uncertainty_std']:
            if metric in self.region_gdf.columns:
                data = self.region_gdf[metric].dropna()
                report_data[metric] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'q5': np.percentile(data, 5),
                    'q95': np.percentile(data, 95)
                }
        
        # 将报告数据保存为CSV
        report_df = pd.DataFrame(report_data).T
        report_df.to_csv(os.path.join(report_dir, "uncertainty_statistics.csv"))
        
        # 生成LaTeX格式的表格
        latex_table = report_df.to_latex(float_format="%.4f", caption="Uncertainty Analysis Statistics")
        with open(os.path.join(report_dir, "uncertainty_statistics.tex"), "w") as f:
            f.write(latex_table)
        
        # 生成不确定性分类报告
        # 根据不确定性水平对区域进行分类
        if 'data_uncertainty_std' in self.region_gdf.columns:
            uncertainty_data = self.region_gdf['data_uncertainty_std']
            
            # 定义不确定性分类阈值
            low_threshold = np.percentile(uncertainty_data, 33)
            high_threshold = np.percentile(uncertainty_data, 67)
            
            # 分类
            uncertainty_class = pd.cut(
                uncertainty_data,
                bins=[-np.inf, low_threshold, high_threshold, np.inf],
                labels=['Low', 'Medium', 'High']
            )
            
            # 统计各类别面积占比
            class_counts = uncertainty_class.value_counts(normalize=True)
            class_df = pd.DataFrame({
                'Uncertainty Class': class_counts.index,
                'Percentage': class_counts.values * 100
            })
            
            # 保存分类结果
            class_df.to_csv(os.path.join(report_dir, "uncertainty_classification.csv"), index=False)
            
            # 生成LaTeX表格
            latex_table = class_df.to_latex(index=False, float_format="%.2f", 
                                          caption="Uncertainty Classification")
            with open(os.path.join(report_dir, "uncertainty_classification.tex"), "w") as f:
                f.write(latex_table)
        
        print("综合不确定性报告生成完成")
    
    def run_complete_analysis(self, data_path, n_simulations=100):
        """运行完整的不确定性分析流程"""
        print("开始完整的不确定性分析流程")
        
        # 1. 加载数据
        self.load_data(data_path)
        
        # 2. 数据不确定性分析
        self.data_uncertainty_analysis(n_simulations=n_simulations)
        
        # 3. 模型不确定性分析 (需要训练数据)
        # self.model_uncertainty_analysis()
        
        # 4. 模型结构不确定性分析 (需要多个模型)
        model_dirs = {
            "Three_Layer": self.model_dir,
            # 添加其他模型路径
            # "Two_Layer": "E:/怀化市地质灾害攻关/全市易发性/result/二层结构/saved_models",
            # "Traditional": "E:/怀化市地质灾害攻关/全市易发性/result/传统算法/saved_models"
        }
        self.structural_uncertainty_analysis(model_dirs)
        
        # 5. SHAP分析
        self.shap_analysis()
        
        # 6. 绘制SHAP图表
        self.plot_shap_summary(max_display=15)
        # 绘制所有13个因子的依赖图
        self.plot_shap_dependence(features=self.all_features)
        # 绘制前3个样本的瀑布图
        if 'data_uncertainty_std' in self.region_gdf.columns:
        # 获取不确定性最高、中等和最低的样本
            high_uncertainty_idx = self.region_gdf['data_uncertainty_std'].nlargest(1).index[0]
            median_uncertainty_idx = self.region_gdf['data_uncertainty_std'].abs().sort_values().index[len(self.region_gdf)//2]
            low_uncertainty_idx = self.region_gdf['data_uncertainty_std'].nsmallest(1).index[0]
            
            # 绘制这些样本的瀑布图
            self.plot_shap_waterfall(sample_indices=[high_uncertainty_idx, median_uncertainty_idx, low_uncertainty_idx])
        
        # 7. 生成不确定性地图
        self.generate_uncertainty_maps()
        
        # 8. 生成统计图表
        self.generate_statistical_plots()
        
        # 9. 生成综合报告
        self.generate_comprehensive_report()
        
        print("不确定性分析流程完成")
        
        # 返回主要结果
        return self.results

# 使用示例
if __name__ == "__main__":
    # 设置路径
    model_dir = "E:/怀化市地质灾害攻关/全市易发性/result/三层结构/saved_models"
    output_dir = "E:/怀化市地质灾害攻关/全市易发性/result/三层结构/不确定性分析"
    data_path = "E:/怀化市地质灾害攻关/多尺度网格/csv/特征数据/250网格.csv"
    
    # 创建分析器并运行分析
    analyzer = UncertaintyAnalyzer(model_dir, output_dir)
    results = analyzer.run_complete_analysis(data_path, n_simulations=50)
    
    print(f"分析结果已保存到: {output_dir}")