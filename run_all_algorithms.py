# -*- coding: utf-8 -*-
"""
主运行脚本，用于运行所有三种Stacking算法并生成比较报告
"""

import os
import logging
import json
import matplotlib
# 设置matplotlib使用非交互式后端，避免Tkinter线程问题
matplotlib.use('Agg')  # 必须在导入其他matplotlib模块之前设置

from data_processor import UnifiedDataProcessor
from model_evaluator import PaperReadyEvaluator

from Stacking import Config as StackingConfig, StackingEnsemble, ReportGenerator
from 模型组合模块.Stackingtechnology import Config as TwoLayerConfig, EnsemblePipeline
from three_layer_Stacking import Config as ThreeLayerConfig, MultiLayerEnsemblePipeline
            
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("algorithm_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_algorithm(algorithm_name, data_path, output_root):
    """运行单个算法"""
    logger.info(f"开始运行 {algorithm_name}")
    
    try:
        # 导入算法模块
        if algorithm_name == "Traditional_Stacking":
            # 更新输出路径
            StackingConfig.OUTPUT_ROOT = os.path.join(output_root, "传统算法")
            os.makedirs(StackingConfig.OUTPUT_ROOT, exist_ok=True)
            
            # 数据预处理
            processor = UnifiedDataProcessor(
                target_col=StackingConfig.TARGET_COL,
                test_size=StackingConfig.TEST_SIZE,
                random_state=StackingConfig.RANDOM_STATE,
                feature_selection=StackingConfig.FEATURE_SELECTION,
                balance_method=StackingConfig.BALANCE_METHOD,
                exclude_columns=['lon', 'lat']
            )
            
            # 指定VIF和容忍度输出目录
            vif_output_dir = os.path.join(output_root, "data_diagnostics", "传统算法")
            X, y = processor.load_and_preprocess(data_path, output_dir=vif_output_dir)
            
            # 训练模型
            ensemble = StackingEnsemble(X, y)
            results = ensemble.train()
            ensemble.save_models()
            
            # 生成报告
            report = ReportGenerator.generate_report(ensemble, results, processor, data_path)
            
        elif algorithm_name == "Two_Layer_Stacking":
            # 更新输出路径
            TwoLayerConfig.OUTPUT_ROOT = os.path.join(output_root, "二层结构")
            os.makedirs(TwoLayerConfig.OUTPUT_ROOT, exist_ok=True)
            
            # 数据预处理
            processor = UnifiedDataProcessor(
                target_col=TwoLayerConfig.TARGET_COL,
                test_size=TwoLayerConfig.TEST_SIZE,
                random_state=TwoLayerConfig.RANDOM_STATE,
                feature_selection=TwoLayerConfig.FEATURE_SELECTION,
                balance_method=TwoLayerConfig.BALANCE_METHOD,
                exclude_columns=['lon', 'lat']
            )
            
            # 指定VIF和容忍度输出目录
            vif_output_dir = os.path.join(output_root, "data_diagnostics", "二层结构")
            X, y = processor.load_and_preprocess(data_path, output_dir=vif_output_dir)
            
            # 训练模型
            pipeline = EnsemblePipeline(X, y, scaler=processor.scaler, 
                                      selected_features=processor.selected_features)
            results = pipeline.train()
            pipeline.save_models()
            
            # 保存评估报告
            report_path = os.path.join(TwoLayerConfig.OUTPUT_ROOT, "evaluation_report.json")
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=4)
            
        elif algorithm_name == "Three_Layer_Stacking":
            # 更新输出路径
            ThreeLayerConfig.OUTPUT_ROOT = os.path.join(output_root, "三层结构")
            os.makedirs(ThreeLayerConfig.OUTPUT_ROOT, exist_ok=True)
            
            # 数据预处理
            processor = UnifiedDataProcessor(
                target_col=ThreeLayerConfig.TARGET_COL,
                test_size=ThreeLayerConfig.TEST_SIZE,
                random_state=ThreeLayerConfig.RANDOM_STATE,
                feature_selection=ThreeLayerConfig.FEATURE_SELECTION,
                balance_method=ThreeLayerConfig.BALANCE_METHOD,
                exclude_columns=['lon', 'lat']
            )
            
            # 指定VIF和容忍度输出目录
            vif_output_dir = os.path.join(output_root, "data_diagnostics", "三层结构")
            X, y = processor.load_and_preprocess(data_path, output_dir=vif_output_dir)
            
            # 训练模型
            pipeline = MultiLayerEnsemblePipeline(X, y, scaler=processor.scaler,
                                                selected_features=processor.selected_features)
            results = pipeline.train()
            pipeline.save_models()
            
            # 保存评估报告
            report_path = os.path.join(ThreeLayerConfig.OUTPUT_ROOT, "evaluation_report.json")
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=4)
        else:
            raise ValueError(f"未知算法: {algorithm_name}")
        
        logger.info(f"{algorithm_name} 完成训练")
        return results
        
    except Exception as e:
        logger.error(f"{algorithm_name} 运行失败: {str(e)}")
        raise

def main():
    DATA_PATH = "E:/怀化市地质灾害攻关/全市易发性/样本/样本数据_CF.csv"
    OUTPUT_ROOT = "E:/怀化市地质灾害攻关/全市易发性/result/算法比较910"
    
    # 创建评估器
    evaluator = PaperReadyEvaluator(OUTPUT_ROOT)
    
    # 运行三种算法前先进行数据诊断
    processor = UnifiedDataProcessor(
        target_col='target',
        test_size=0.2,
        random_state=42,
        feature_selection=True,
        balance_method='SMOTE',
        exclude_columns=['lon', 'lat']  # 排除经纬度字段
    )
    
    # 加载数据并生成诊断图表
    data_diagnostics_dir = os.path.join(OUTPUT_ROOT, "data_diagnostics")
    X, y = processor.load_and_preprocess(DATA_PATH, output_dir=data_diagnostics_dir)
    processor.generate_diagnostic_plots(data_diagnostics_dir)
    
    # 运行三种算法
    algorithms = [
        ("Traditional_Stacking", "传统Stacking算法"),
        ("Two_Layer_Stacking", "二层Stacking算法"), 
        ("Three_Layer_Stacking", "三层Stacking算法")
    ]
    
    all_results = {}
    for algo_name, algo_display_name in algorithms:
        results = run_algorithm(algo_name, DATA_PATH, OUTPUT_ROOT) 
        all_results[algo_name] = results
    
    # 生成比较图表和报告
    evaluator.results = all_results
    evaluator.generate_algorithm_comparison_plots()
    evaluator.save_comparison_report()
    
    logger.info("所有算法运行完成，比较报告已生成")

if __name__ == "__main__":
    main()