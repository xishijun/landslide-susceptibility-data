# -*- coding: utf-8 -*-
"""
三层Stacking模型预测器
基于已训练的三层Stacking模型结构，实现对新数据的预测功能
包含初级学习层、中级学习层和元学习层的完整三层结构
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Any, Union
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreeLayerStackingPredictor:
    """三层Stacking模型预测器"""
    
    def __init__(self, model_dir: str, target_crs: str = "EPSG:4326"):
        """初始化预测器
        
        Args:
            model_dir: 模型保存目录
            target_crs: 目标坐标系
        """
        self.model_dir = model_dir
        self.target_crs = target_crs
        
        # 初始化各层模型和权重
        self.layer1_models = []  # 初级学习层模型
        self.layer2_models = []  # 中级学习层模型
        self.layer3_models = []  # 元学习层模型
        self.final_meta_model = None  # 最终元学习器
        
        self.scaler = None  # 数据标准化器
        self.feature_columns = []  # 特征列
        self.layer1_weights = []  # 第一层模型权重
        self.layer2_weights = []  # 第二层模型权重
        
        # 配置参数（与训练时保持一致）
        self.use_original_features = True  # 是否在元特征中加入原始特征
        self.weighted_stacking = True  # 是否使用加权Stacking
        
        # 加载模型和配置
        self._load_models()
    
    def _load_models(self):
        """加载已训练的模型和配置"""
        logger.info("开始加载三层Stacking模型...")
        
        # 加载预处理工具
        try:
            with open(os.path.join(self.model_dir, "scaler.pkl"), 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("标准化器加载成功")
        except Exception as e:
            logger.error(f"加载标准化器失败: {e}")
            self.scaler = None
        
        # 加载特征列
        try:
            with open(os.path.join(self.model_dir, "feature_columns.pkl"), 'rb') as f:
                self.feature_columns = pickle.load(f)
            logger.info(f"特征列加载成功: {len(self.feature_columns)}个特征")
        except Exception as e:
            logger.error(f"加载特征列失败: {e}")
            self.feature_columns = []
        
        # 加载模型权重
        try:
            with open(os.path.join(self.model_dir, "model_weights.json"), 'r') as f:
                weights_data = json.load(f)
                self.layer1_weights = weights_data.get('layer1_weights', [])
                self.layer2_weights = weights_data.get('layer2_weights', [])
            logger.info(f"模型权重加载成功: 第一层{len(self.layer1_weights)}个, 第二层{len(self.layer2_weights)}个")
        except Exception as e:
            logger.error(f"加载模型权重失败: {e}")
            self.layer1_weights = []
            self.layer2_weights = []
        
        # 加载第一层模型（初级学习层）
        layer1_files = [f for f in os.listdir(self.model_dir) if f.startswith('layer1_') and f.endswith('.pkl')]
        for model_file in tqdm(layer1_files, desc="加载第一层模型"):
            try:
                model_path = os.path.join(self.model_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.layer1_models.append({
                        'name': model_data['metadata']['name'],
                        'model': model_data['model'],
                        'weight': model_data['metadata'].get('weight', 1.0)
                    })
            except Exception as e:
                logger.error(f"加载第一层模型 {model_file} 失败: {e}")
        logger.info(f"第一层模型加载完成: {len(self.layer1_models)}个模型")
        
        # 加载第二层模型（中级学习层）
        layer2_files = [f for f in os.listdir(self.model_dir) if f.startswith('layer2_') and f.endswith('.pkl')]
        for model_file in tqdm(layer2_files, desc="加载第二层模型"):
            try:
                model_path = os.path.join(self.model_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.layer2_models.append({
                        'name': model_data['metadata']['name'],
                        'model': model_data['model'],
                        'weight': model_data['metadata'].get('weight', 1.0)
                    })
            except Exception as e:
                logger.error(f"加载第二层模型 {model_file} 失败: {e}")
        logger.info(f"第二层模型加载完成: {len(self.layer2_models)}个模型")
        
        # 加载第三层模型（元学习层）
        layer3_files = [f for f in os.listdir(self.model_dir) if f.startswith('layer3_') and f.endswith('.pkl')]
        for model_file in tqdm(layer3_files, desc="加载第三层模型"):
            try:
                model_path = os.path.join(self.model_dir, model_file)
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.layer3_models.append({
                        'name': model_data['metadata']['name'],
                        'model': model_data['model'],
                        'weight': model_data['metadata'].get('weight', 1.0)
                    })
            except Exception as e:
                logger.error(f"加载第三层模型 {model_file} 失败: {e}")
        logger.info(f"第三层模型加载完成: {len(self.layer3_models)}个模型")
        
        # 加载最终元模型
        try:
            with open(os.path.join(self.model_dir, "final_meta_model.pkl"), 'rb') as f:
                self.final_meta_model = pickle.load(f)
            logger.info("最终元模型加载成功")
        except Exception as e:
            logger.error(f"加载最终元模型失败: {e}")
            self.final_meta_model = None
        
        # 加载训练配置
        try:
            with open(os.path.join(self.model_dir, "training_config.json"), 'r') as f:
                training_config = json.load(f)
                self.use_original_features = training_config.get('use_original_features', True)
                self.weighted_stacking = training_config.get('weighted_stacking', True)
            logger.info("训练配置加载成功")
        except Exception as e:
            logger.error(f"加载训练配置失败: {e}")
        
        logger.info(f"模型加载完成: {len(self.layer1_models)}个第一层模型, {len(self.layer2_models)}个第二层模型, {len(self.layer3_models)}个第三层模型, 1个最终元模型")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理
        
        Args:
            data: 输入数据
            
        Returns:
            预处理后的数据
        """
        # 选择特征列
        if self.feature_columns:
            available_features = [col for col in self.feature_columns if col in data.columns]
            if len(available_features) < len(self.feature_columns):
                logger.warning(f"输入数据缺少部分特征: {set(self.feature_columns) - set(available_features)}")
            data = data[available_features]
        else:
            logger.warning("没有可用的特征列信息，使用所有数值列")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            data = data[numeric_cols]
        
        # 数据标准化
        if self.scaler is not None:
            try:
                data = pd.DataFrame(
                    self.scaler.transform(data),
                    columns=data.columns,
                    index=data.index
                )
                logger.info("数据标准化完成")
            except Exception as e:
                logger.error(f"数据标准化失败: {e}")
        
        return data
    
    def _build_meta_features(self, X: pd.DataFrame, preds: List[np.ndarray], 
                           weights: List[float], include_original: bool = True) -> np.ndarray:
        """构建元特征
        
        Args:
            X: 原始特征
            preds: 预测结果列表
            weights: 模型权重列表
            include_original: 是否包含原始特征
            
        Returns:
            元特征矩阵
        """
        meta_features = []
        
        # 添加原始特征（可选）
        if include_original and X is not None:
            meta_features.append(X.values)
        
        # 添加加权预测特征
        for i, pred in enumerate(preds):
            if self.weighted_stacking and weights and i < len(weights):
                weighted_pred = pred * weights[i]
                meta_features.append(weighted_pred.reshape(-1, 1))
            else:
                meta_features.append(pred.reshape(-1, 1))
        
        # 合并所有特征
        return np.hstack(meta_features)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率（三层Stacking结构）
        
        Args:
            X: 输入特征数据
            
        Returns:
            预测概率
        """
        if not self.layer1_models or not self.layer2_models or self.final_meta_model is None:
            raise ValueError("模型未正确加载，无法进行预测")
        
        # 第一层预测（初级学习层）
        logger.info("进行第一层预测...")
        layer1_preds = []
        for model_info in tqdm(self.layer1_models, desc="第一层预测"):
            try:
                pred = model_info['model'].predict_proba(X)[:, 1]
                layer1_preds.append(pred)
            except Exception as e:
                logger.error(f"第一层模型 {model_info['name']} 预测失败: {e}")
                # 预测失败时填充默认值
                layer1_preds.append(np.zeros(X.shape[0]))
        
        # 构建第二层输入特征（初级预测结果 + 原始特征）
        logger.info("构建第二层输入特征...")
        layer1_meta_features = self._build_meta_features(
            X, layer1_preds, self.layer1_weights, self.use_original_features
        )
        
        # 第二层预测（中级学习层）
        logger.info("进行第二层预测...")
        layer2_preds = []
        for model_info in tqdm(self.layer2_models, desc="第二层预测"):
            try:
                pred = model_info['model'].predict_proba(layer1_meta_features)[:, 1]
                layer2_preds.append(pred)
            except Exception as e:
                logger.error(f"第二层模型 {model_info['name']} 预测失败: {e}")
                # 预测失败时填充默认值
                layer2_preds.append(np.zeros(X.shape[0]))
        
        # 构建最终元特征（第二层预测结果，不包括原始特征）
        logger.info("构建最终元特征...")
        final_meta_features = self._build_meta_features(
            None, layer2_preds, self.layer2_weights, include_original=False
        )
        
        # 最终元模型预测
        logger.info("进行最终元模型预测...")
        try:
            y_proba = self.final_meta_model.predict_proba(final_meta_features)[:, 1]
            return y_proba
        except Exception as e:
            logger.error(f"最终元模型预测失败: {e}")
            # 如果元模型预测失败，使用第二层模型的加权平均
            weighted_preds = np.zeros(X.shape[0])
            for i, pred in enumerate(layer2_preds):
                weight = self.layer2_weights[i] if i < len(self.layer2_weights) else 1.0
                weighted_preds += pred * weight
            
            # 归一化到[0, 1]区间
            if self.layer2_weights:
                weighted_preds /= sum(self.layer2_weights)
            else:
                weighted_preds /= len(layer2_preds)
            
            return weighted_preds
    
    def predict_risk(self, input_path: str, output_path: str, batch_size: int = 1000) -> gpd.GeoDataFrame:
        """执行区域风险预测
        
        Args:
            input_path: 输入数据路径
            output_path: 输出数据路径
            batch_size: 批处理大小（提高大文件处理效率）
            
        Returns:
            包含预测结果的地理数据框
        """
        logger.info(f"正在加载输入数据: {input_path}")
        
        # 分块读取大文件
        chunks = []
        for chunk in tqdm(pd.read_csv(input_path, chunksize=batch_size), desc="读取数据"):
            chunks.append(chunk)
        
        region_data = pd.concat(chunks, ignore_index=True)
        logger.info(f"已加载输入数据，共 {len(region_data)} 条记录")
        
        # 检查必要的字段是否存在
        required_columns = ['lon', 'lat']
        if not all(col in region_data.columns for col in required_columns):
            raise ValueError(f"输入数据缺少必要的列: {required_columns}")
        
        # 创建地理数据框架
        region_gdf = gpd.GeoDataFrame(
            region_data,
            geometry=gpd.points_from_xy(region_data.lon, region_data.lat),
            crs=self.target_crs
        )
        
        # 数据预处理
        logger.info("正在进行数据预处理...")
        processed_data = self.preprocess_data(region_data)
        
        # 使用三层stacking模型进行预测
        logger.info("使用三层stacking模型进行预测...")
        try:
            # 直接使用stacking模型预测概率
            probabilities = self.predict_proba(processed_data)
            region_gdf['probability'] = probabilities
        except Exception as e:
            logger.error(f"批量预测错误: {e}")
            logger.info("尝试使用逐样本预测...")
            
            # 逐样本预测（效率较低，仅作为备选方案）
            probabilities = []
            for idx, row in tqdm(processed_data.iterrows(), total=len(processed_data), desc="逐样本预测"):
                try:
                    # 单样本预测
                    row_df = pd.DataFrame([row])
                    
                    # 第一层预测
                    layer1_preds = []
                    for model_info in self.layer1_models:
                        pred = model_info['model'].predict_proba(row_df)[:, 1]
                        layer1_preds.append(pred[0])
                    
                    # 构建第二层输入
                    layer1_meta = self._build_meta_features(
                        row_df, [np.array([p]) for p in layer1_preds], 
                        self.layer1_weights, self.use_original_features
                    )
                    
                    # 第二层预测
                    layer2_preds = []
                    for model_info in self.layer2_models:
                        pred = model_info['model'].predict_proba(layer1_meta)[:, 1]
                        layer2_preds.append(pred[0])
                    
                    # 构建最终元特征
                    final_meta = self._build_meta_features(
                        None, [np.array([p]) for p in layer2_preds], 
                        self.layer2_weights, include_original=False
                    )
                    
                    # 最终预测
                    prob = self.final_meta_model.predict_proba(final_meta)[0, 1]
                    probabilities.append(prob)
                except Exception as inner_e:
                    logger.error(f"单样本预测失败: {inner_e}")
                    probabilities.append(0)  # 预测失败时填充默认值
            region_gdf['probability'] = probabilities
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存预测结果
        logger.info(f"保存预测结果到: {output_path}")
        region_gdf[['lon', 'lat', 'probability']].to_csv(output_path, index=False)
        
        # 同时保存为GeoPackage格式（可选）
        try:
            gpkg_path = output_path.replace('.csv', '.gpkg')
            region_gdf.to_file(gpkg_path, driver='GPKG')
            logger.info(f"地理数据已保存为: {gpkg_path}")
        except Exception as e:
            logger.error(f"保存GeoPackage失败: {e}")
        
        return region_gdf

# 使用示例
if __name__ == "__main__":
    # 设置模型目录和输入输出路径
    model_dir = "E:/怀化市地质灾害攻关/全市易发性/result/三层结构/saved_models"
    input_path = "E:/怀化市地质灾害攻关/多尺度网格/csv/特征数据/100网格.csv"
    output_path = "E:/怀化市地质灾害攻关/全市易发性/result/三层结构/预测结果/三层结构预测结果100网格.csv"
    
    # 创建预测器并执行预测
    predictor = ThreeLayerStackingPredictor(model_dir)
    result_gdf = predictor.predict_risk(input_path, output_path)
    
    print(f"预测完成，结果已保存到: {output_path}")
    print(f"预测概率范围: {result_gdf['probability'].min():.4f} - {result_gdf['probability'].max():.4f}")