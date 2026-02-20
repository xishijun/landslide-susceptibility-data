# -*- coding: utf-8 -*-
"""
生成论文级质量评估图表的评估器
支持三种Stacking算法的对比分析
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
    brier_score_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, log_loss, mean_squared_error, mean_absolute_error
)
from sklearn.calibration import calibration_curve
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PaperReadyEvaluator:
    """生成论文级质量评估图表的评估器"""
    
    # 为三种算法定义不同的颜色和样式
    ALGORITHM_STYLES = {
        'Traditional_Stacking': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},  # 蓝色
        'Two_Layer_Stacking': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's'},   # 橙色
        'Three_Layer_Stacking': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^'}   # 绿色
    }
    
    def __init__(self, output_root, figure_size=(10, 8), dpi=300):
        self.output_root = output_root
        self.figure_size = figure_size
        self.dpi = dpi
        self.results = {}
        
        # 设置Catena期刊风格的图表
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5
        plt.rcParams['legend.frameon'] = False
        
    def evaluate_classification(self, y_true, y_pred, y_proba, algorithm_name):
        """全面评估分类模型性能"""
        # 计算二分类问题的概率预测
        y_proba_binary = y_proba
        
        # 计算所有指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),  # OA (Overall Accuracy)
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),  # AUC
            'pr_auc': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'mcc': matthews_corrcoef(y_true, y_pred),  # MCC
            'kappa': cohen_kappa_score(y_true, y_pred),  # Kappa
            'rmse': np.sqrt(mean_squared_error(y_true, y_proba_binary)),  # RMSE
            'mae': mean_absolute_error(y_true, y_proba_binary)  # MAE
        }
        
        # 计算每个类别的精确率、召回率和F1分数
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'confusion_matrix': cm.tolist(),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,  # 阴性预测值
        })
        
        # 保存ROC和PR曲线数据用于后续比较
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        metrics['roc_curve'] = (fpr.tolist(), tpr.tolist())
        metrics['pr_curve'] = (recall.tolist(), precision.tolist())
        
        # 保存结果
        self.results[algorithm_name] = metrics
        
        return metrics
    
    def generate_algorithm_plots(self, y_true, y_proba, y_pred, output_dir, algorithm_name):
        """生成单个算法的评估图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=self.figure_size)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algorithm_name}_roc_curve.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # 2. PR曲线
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=self.figure_size)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curve', fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algorithm_name}_pr_curve.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # 3. 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=self.figure_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Non-Landslide', 'Landslide'],
                   yticklabels=['Non-Landslide', 'Landslide'])
        plt.title('Confusion Matrix', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algorithm_name}_confusion_matrix.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # 4. 校准曲线
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        plt.figure(figsize=self.figure_size)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', linewidth=2)
        plt.xlabel('Predicted Probability', fontweight='bold')
        plt.ylabel('True Probability', fontweight='bold')
        plt.title('Calibration Curve', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algorithm_name}_calibration_curve.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # 5. 预测概率分布
        plt.figure(figsize=self.figure_size)
        plt.hist(y_proba[y_true == 0], bins=30, alpha=0.5, label='Negative Class', color='blue')
        plt.hist(y_proba[y_true == 1], bins=30, alpha=0.5, label='Positive Class', color='red')
        plt.xlabel('Predicted Probability', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title('Prediction Probability Distribution', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algorithm_name}_probability_distribution.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # 6. 性能指标雷达图（单个算法）
        self._create_single_algorithm_radar(output_dir, algorithm_name)
    
    def _create_single_algorithm_radar(self, output_dir, algorithm_name):
        """为单个算法创建性能指标雷达图"""
        if algorithm_name not in self.results:
            return
            
        metrics = self.results[algorithm_name]
        
        # 选择要展示的指标
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']
        metrics_labels = ['OA', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
        
        values = [metrics[metric] for metric in metrics_to_show]
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics_to_show), endpoint=False).tolist()
        values = np.array(values)
        values = np.concatenate((values, [values[0]]))  # 闭合雷达图
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        style = self.ALGORITHM_STYLES.get(algorithm_name, {})
        ax.plot(angles, values, 
               color=style.get('color', 'blue'),
               linestyle=style.get('linestyle', '-'),
               linewidth=2,
               label=algorithm_name)
        ax.fill(angles, values, alpha=0.1, color=style.get('color', 'blue'))
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_title(f'{algorithm_name} Performance Radar', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{algorithm_name}_performance_radar.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def generate_algorithm_comparison_plots(self):
        """生成三种算法的比较图表"""
        figures_dir = os.path.join(self.output_root, "comparison_figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # 1. ROC曲线比较
        plt.figure(figsize=self.figure_size)
        for algo_name, metrics in self.results.items():
            if 'roc_curve' in metrics:
                fpr, tpr = metrics['roc_curve']
                style = self.ALGORITHM_STYLES.get(algo_name, {})
                plt.plot(fpr, tpr, 
                        color=style.get('color', 'blue'),
                        linestyle=style.get('linestyle', '-'),
                        marker=style.get('marker', None),
                        markevery=0.1,
                        label=f'{algo_name} (AUC = {metrics["roc_auc"]:.3f})',
                        linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curve Comparison of Three Stacking Algorithms', fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "roc_curve_comparison.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # 2. PR曲线比较
        plt.figure(figsize=self.figure_size)
        for algo_name, metrics in self.results.items():
            if 'pr_curve' in metrics:
                recall, precision = metrics['pr_curve']
                style = self.ALGORITHM_STYLES.get(algo_name, {})
                plt.plot(recall, precision,
                        color=style.get('color', 'blue'),
                        linestyle=style.get('linestyle', '-'),
                        marker=style.get('marker', None),
                        markevery=0.1,
                        label=f'{algo_name} (AP = {metrics["pr_auc"]:.3f})',
                        linewidth=2)
        
        plt.xlabel('Recall', fontweight='bold')
        plt.ylabel('Precision', fontweight='bold')
        plt.title('Precision-Recall Curve Comparison', fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "pr_curve_comparison.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # 3. 性能指标雷达图
        self._create_radar_chart(figures_dir)
        
        # 4. 性能指标柱状图比较
        self._create_metric_bar_chart(figures_dir)
        
        # 5. 综合指标对比表
        self._create_comprehensive_comparison_table(figures_dir)
        
        logger.info(f"算法比较图表已保存至: {figures_dir}")
    
    def _create_radar_chart(self, output_dir):
        """创建性能指标雷达图"""
        # 选择要展示的指标
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'mcc']
        metrics_labels = ['OA', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
        
        # 准备数据
        algorithms = list(self.results.keys())
        values = []
        for algo in algorithms:
            algo_values = [self.results[algo][metric] for metric in metrics_to_show]
            values.append(algo_values)
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics_to_show), endpoint=False).tolist()
        values = np.array(values)
        values = np.concatenate((values, values[:,[0]]), axis=1)
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=self.figure_size, subplot_kw=dict(polar=True))
        
        for i, algo in enumerate(algorithms):
            style = self.ALGORITHM_STYLES.get(algo, {})
            ax.plot(angles, values[i], 
                   color=style.get('color', 'blue'),
                   linestyle=style.get('linestyle', '-'),
                   linewidth=2,
                   label=algo)
            ax.fill(angles, values[i], alpha=0.1, color=style.get('color', 'blue'))
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_title('Performance Metrics Radar Chart', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_radar.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_metric_bar_chart(self, output_dir):
        """创建性能指标柱状图比较"""
        metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'mcc', 'kappa']
        metrics_labels = ['OA', 'Precision', 'Recall', 'F1', 'AUC', 'AP', 'MCC', 'Kappa']
        
        algorithms = list(self.results.keys())
        x = np.arange(len(metrics_to_show))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i, algo in enumerate(algorithms):
            values = [self.results[algo][metric] for metric in metrics_to_show]
            style = self.ALGORITHM_STYLES.get(algo, {})
            ax.bar(x + i*width, values, width, 
                  color=style.get('color', 'blue'),
                  label=algo, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_bar_chart.png"), 
                   dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_comprehensive_comparison_table(self, output_dir):
        """创建综合指标对比表"""
        # 选择要展示的指标
        metrics_to_show = [
            'accuracy', 'precision', 'recall', 'f1', 
            'roc_auc', 'pr_auc', 'mcc', 'kappa', 
            'rmse', 'mae', 'log_loss', 'brier_score'
        ]
        
        metrics_labels = [
            'OA', 'Precision', 'Recall', 'F1', 
            'AUC', 'AP', 'MCC', 'Kappa', 
            'RMSE', 'MAE', 'Log Loss', 'Brier Score'
        ]
        
        # 创建数据框
        comparison_data = []
        for algo_name, metrics in self.results.items():
            row = [algo_name]
            for metric in metrics_to_show:
                if metric in metrics:
                    # 对于误差指标，保留更多小数位
                    if metric in ['rmse', 'mae', 'log_loss', 'brier_score']:
                        row.append(f"{metrics[metric]:.4f}")
                    else:
                        row.append(f"{metrics[metric]:.3f}")
                else:
                    row.append("N/A")
            comparison_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(comparison_data, columns=['Algorithm'] + metrics_labels)
        
        # 保存为CSV和LaTeX表格格式
        df.to_csv(os.path.join(output_dir, "comprehensive_comparison.csv"), index=False)
        
        # 生成LaTeX表格
        latex_table = df.to_latex(index=False, float_format="%.3f")
        with open(os.path.join(output_dir, "comprehensive_comparison.tex"), "w") as f:
            f.write(latex_table)
        
        logger.info("综合指标对比表已保存")
    
    def save_comparison_report(self):
        """保存算法比较报告"""
        report_path = os.path.join(self.output_root, "algorithm_comparison_report.json")
        
        # 转换为可序列化的格式
        serializable_results = {}
        for algo, metrics in self.results.items():
            serializable_results[algo] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                         for k, v in metrics.items()}
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"算法比较报告已保存至: {report_path}")
        return serializable_results