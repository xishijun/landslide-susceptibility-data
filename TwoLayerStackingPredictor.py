# -*- coding: utf-8 -*-
"""
二层Stacking融合型滑坡预测模型 - 预测器
用于加载已训练好的模型并对新数据进行预测
"""
import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple, Optional
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TwoLayerStackingPredictor:
    """二层Stacking模型预测器"""
    
    def __init__(self, model_dir: str, target_crs: str = "EPSG:4326"):
        """初始化预测器
        
        Args:
            model_dir: 模型文件目录路径
            target_crs: 目标坐标系
        """
        self.model_dir = model_dir
        self.target_crs = target_crs
        self.base_models = {}  # 基学习器字典
        self.meta_model = None  # 元学习器
        self.scaler = None  # 数据标准化器
        self.feature_columns = None  # 特征列名
        self.base_model_names = ['Logistic', 'SVM', 'RandomForest', 'AdaBoost', 'XGBoost']
        
    def load_models(self) -> None:
        """加载所有已训练的模型"""
        logger.info("开始加载模型...")
        
        try:
            # 加载基学习器
            for name in self.base_model_names:
                model_path = os.path.join(self.model_dir, f"{name}_base.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        # 根据Stackingtechnology.py的保存格式，模型在'model'键中
                        if 'model' in model_data:
                            self.base_models[name] = model_data['model']
                        else:
                            # 如果直接保存的是模型对象
                            self.base_models[name] = model_data
                    logger.info(f"已加载基学习器: {name}")
                else:
                    logger.warning(f"基学习器文件不存在: {model_path}")
            
            # 加载元学习器
            meta_model_path = os.path.join(self.model_dir, "meta_model.pkl")
            if os.path.exists(meta_model_path):
                with open(meta_model_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                logger.info("已加载元学习器")
            else:
                logger.warning(f"元学习器文件不存在: {meta_model_path}")
            
            # 加载标准化器
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("已加载标准化器")
            else:
                logger.warning(f"标准化器文件不存在: {scaler_path}")
            
            # 加载特征列名
            feature_cols_path = os.path.join(self.model_dir, "feature_columns.pkl")
            if os.path.exists(feature_cols_path):
                with open(feature_cols_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logger.info("已加载特征列名")
            else:
                logger.warning(f"特征列名文件不存在: {feature_cols_path}")
                
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据（特征选择、标准化）
        
        Args:
            data: 原始数据
            
        Returns:
            预处理后的数据
        """
        logger.info("开始预处理数据...")
        
        # 检查特征列是否存在
        if self.feature_columns is None:
            logger.error("特征列名为空，无法进行特征选择")
            raise ValueError("特征列名为空")
        
        # 选择特征
        selected_data = data[self.feature_columns].copy()
        logger.info(f"特征选择完成，选择 {len(self.feature_columns)} 个特征")
        
        # 数据标准化
        if self.scaler is not None:
            scaled_data = self.scaler.transform(selected_data)
            processed_data = pd.DataFrame(scaled_data, columns=self.feature_columns)
            logger.info("数据标准化完成")
        else:
            processed_data = selected_data
            logger.warning("标准化器为空，跳过标准化步骤")
        
        return processed_data
    
    def _build_meta_features(self, X: pd.DataFrame, include_original: bool = True) -> np.ndarray:
        """构建元特征矩阵（核心方法，体现Stacking设计思想）
        
        Args:
            X: 输入特征数据
            include_original: 是否包含原始特征
            
        Returns:
            元特征矩阵
        """
        meta_features = []
        
        # 添加每个基学习器的预测概率（正类的概率）
        for name, model in self.base_models.items():
            try:
                # 获取正类的预测概率
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[:, 1]  # 正类的概率
                else:
                    # 对于某些不支持概率预测的模型，使用决策函数或直接预测
                    logger.warning(f"{name} 模型不支持概率预测，使用决策函数")
                    if hasattr(model, 'decision_function'):
                        decision_values = model.decision_function(X)
                        # 将决策函数值转换为概率（粗略估计）
                        proba = 1 / (1 + np.exp(-decision_values))
                    else:
                        logger.warning(f"{name} 模型不支持概率预测和决策函数，使用二值预测")
                        proba = model.predict(X).astype(float)
                
                meta_features.append(proba.reshape(-1, 1))
                logger.debug(f"已添加 {name} 的预测概率到元特征")
                
            except Exception as e:
                logger.error(f"构建 {name} 的元特征失败: {str(e)}")
                # 添加零向量作为占位符，保持维度一致
                meta_features.append(np.zeros((X.shape[0], 1)))
        
        # 包含原始特征（与训练时保持一致）
        if include_original:
            meta_features.append(X.values if isinstance(X, pd.DataFrame) else X)
            logger.debug("已添加原始特征到元特征")
        
        # 水平拼接所有特征，形成最终的元特征矩阵
        meta_matrix = np.hstack(meta_features)
        logger.info(f"元特征矩阵构建完成，形状: {meta_matrix.shape}")
        
        return meta_matrix
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率
        
        Args:
            X: 输入特征数据
            
        Returns:
            预测概率
        """
        logger.info("开始预测概率...")
        
        # 预处理数据
        processed_X = self.preprocess_data(X)
        
        # 构建元特征（体现Stacking算法的核心思想）
        meta_features = self._build_meta_features(processed_X, include_original=True)
        
        # 使用元学习器进行最终预测
        if self.meta_model is not None:
            # 预测概率
            y_proba = self.meta_model.predict_proba(meta_features)[:, 1]
            logger.info("概率预测完成")
            return y_proba
        else:
            logger.error("元学习器未加载，无法进行预测")
            raise ValueError("元学习器未加载")
    
    def predict_risk(self, input_path: str, output_path: str) -> gpd.GeoDataFrame:
        """执行区域风险预测
        
        Args:
            input_path: 输入数据路径
            output_path: 输出数据路径
            
        Returns:
            包含预测结果的地理数据框
        """
        logger.info(f"正在加载输入数据: {input_path}")
        region_data = pd.read_csv(input_path)
        logger.info(f"已加载输入数据，共 {len(region_data)} 条记录")
        
        # 检查必要的字段是否存在
        required_columns = ['lon', 'lat'] + self.feature_columns
        missing_columns = [col for col in required_columns if col not in region_data.columns]
        
        if missing_columns:
            raise ValueError(f"输入数据缺少必要的列: {missing_columns}")
        
        # 创建地理数据框架
        region_gdf = gpd.GeoDataFrame(
            region_data,
            geometry=gpd.points_from_xy(region_data.lon, region_data.lat),
            crs=self.target_crs
        )
        
        # 坐标系转换（如果需要）
        if region_gdf.crs != self.target_crs:
            logger.info(f"坐标系转换: {region_gdf.crs} -> {self.target_crs}")
            region_gdf = region_gdf.to_crs(self.target_crs)
        
        # 数据预处理
        logger.info("正在进行数据预处理...")
        processed_data = self.preprocess_data(region_data[self.feature_columns])
        
        # 使用stacking模型进行预测
        logger.info("使用stacking模型进行预测...")
        try:
            # 直接使用stacking模型预测概率
            probabilities = self.predict_proba(region_data[self.feature_columns])
            region_gdf['probability'] = probabilities
        except Exception as e:
            logger.error(f"预测错误: {e}")
            # 如果预测失败，尝试使用逐样本预测
            logger.info("尝试使用逐样本预测...")
            probabilities = []
            for _, row in tqdm(processed_data.iterrows(), total=len(processed_data), desc="逐样本预测"):
                try:
                    # 构建单样本的元特征
                    meta_features = self._build_meta_features(pd.DataFrame([row]), include_original=True)
                    prob = self.meta_model.predict_proba(meta_features)[0, 1]
                    probabilities.append(prob)
                except Exception as e:
                    logger.error(f"单样本预测失败: {e}")
                    probabilities.append(0)  # 预测失败时填充默认值
            region_gdf['probability'] = probabilities
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存预测结果
        logger.info(f"保存预测结果到: {output_path}")
        region_gdf[['lon', 'lat', 'probability']].to_csv(output_path, index=False)
        
        # 同时保存为GeoPackage格式（可选）
        try:
            # 设置GDAL_DATA环境变量以避免警告
            if 'GDAL_DATA' not in os.environ:
                # 尝试查找常见的GDAL数据路径
                possible_paths = [
                    os.path.join(os.path.dirname(os.__file__), 'Library', 'share', 'gdal'),
                    os.path.join(os.path.dirname(os.__file__), 'share', 'gdal'),
                    '/usr/share/gdal',
                    '/usr/local/share/gdal'
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        os.environ['GDAL_DATA'] = path
                        logger.info(f"设置 GDAL_DATA = {path}")
                        break
            
            gpkg_path = output_path.replace('.csv', '.gpkg')
            region_gdf.to_file(gpkg_path, driver='GPKG')
            logger.info(f"地理数据已保存为: {gpkg_path}")
        except Exception as e:
            logger.error(f"保存GeoPackage失败: {e}")
        
        return region_gdf


def main():
    """主函数"""
    # 设置路径
    model_dir = "E:/怀化市地质灾害攻关/全市易发性/result/二层结构/saved_models"
    input_path = "E:/怀化市地质灾害攻关/多尺度网格/csv/特征数据/250网格.csv"
    output_path = "E:/怀化市地质灾害攻关/全市易发性/result/二层结构/预测结果/二层结构预测结果250网格.csv"
    
    try:
        logger.info("开始二层Stacking模型预测")
        
        # 初始化预测器
        predictor = TwoLayerStackingPredictor(model_dir)
        0
        # 加载模型
        predictor.load_models()
        
        # 执行区域风险预测
        result_gdf = predictor.predict_risk(input_path, output_path)
        
        # 输出统计信息
        landslide_count = np.sum(result_gdf['probability'] > 0.5)
        landslide_ratio = landslide_count / len(result_gdf) * 100
        logger.info(f"预测结果统计: 总样本数 {len(result_gdf)}，滑坡样本 {landslide_count}，占比 {landslide_ratio:.2f}%")
        
    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()