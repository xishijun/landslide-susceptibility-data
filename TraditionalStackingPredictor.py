import pandas as pd
import numpy as np
import pickle
import os
import geopandas as gpd
from tqdm import tqdm
import warnings
from sklearn.ensemble import StackingClassifier

class TraditionalStackingPredictor:
    def __init__(self, model_dir, input_path, output_path, target_crs="EPSG:4326"):
        self.model_dir = model_dir
        self.input_path = input_path
        self.output_path = output_path
        self.target_crs = target_crs
        self.model = self.load_stacking_model()
        
        # 根据提供的字段设置特征列
        self.feature_columns = [
            'slope', 'landuse', 'panarcurvature', 'profilecurvature', 'NDVI', 'Soil',
            'sratumboundarydistance', 'geomorphictype', 'fracturedistance', 
            'geologicalrockdistance', 'rivernetworkdistance', 'DEM', 'geologicalstructure'
        ]
    
    def load_stacking_model(self):
        """加载stacking模型"""
        stacking_path = os.path.join(self.model_dir, 'stacking_model.pkl')
        
        if not os.path.exists(stacking_path):
            raise FileNotFoundError(f"错误: 未找到stacking模型文件 {stacking_path}")
        
        print("正在加载stacking模型...")
        with open(stacking_path, 'rb') as f:
            model = pickle.load(f)
            
            # 检查模型类型
            if isinstance(model, StackingClassifier):
                print("成功加载StackingClassifier模型")
                return model
            else:
                # 尝试从字典中提取模型
                if isinstance(model, dict) and 'model' in model:
                    print("从字典中提取StackingClassifier模型")
                    return model['model']
                else:
                    raise ValueError("无法识别的模型格式")
    
    def preprocess_data(self, data):
        """数据预处理函数，根据实际情况调整"""
        # 这里可以添加数据清洗、标准化等预处理步骤
        # 示例：填充缺失值
        processed_data = data.fillna(data.mean())
        return processed_data
    
    def predict_risk(self):
        """执行区域风险预测"""
        # 加载输入数据
        print(f"正在加载输入数据: {self.input_path}")
        region_data = pd.read_csv(self.input_path)
        print(f"已加载输入数据，共 {len(region_data)} 条记录")
        
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
            print(f"坐标系转换: {region_gdf.crs} -> {self.target_crs}")
            region_gdf = region_gdf.to_crs(self.target_crs)
        
        # 数据预处理
        print("正在进行数据预处理...")
        processed_data = self.preprocess_data(region_data[self.feature_columns])
        
        # 使用stacking模型进行预测
        print("使用stacking模型进行预测...")
        try:
            # 直接使用stacking模型预测概率
            probabilities = self.model.predict_proba(processed_data)[:, 1]
            region_gdf['probability'] = probabilities
        except Exception as e:
            print(f"预测错误: {e}")
            # 如果预测失败，尝试使用逐样本预测
            print("尝试使用逐样本预测...")
            probabilities = []
            for _, row in tqdm(processed_data.iterrows(), total=len(processed_data), desc="逐样本预测"):
                try:
                    prob = self.model.predict_proba([row])[0, 1]
                    probabilities.append(prob)
                except:
                    probabilities.append(0)  # 预测失败时填充默认值
            region_gdf['probability'] = probabilities
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 保存预测结果
        print(f"保存预测结果到: {self.output_path}")
        region_gdf[['lon', 'lat', 'probability']].to_csv(self.output_path, index=False)
        
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
                        print(f"设置 GDAL_DATA = {path}")
                        break
            
            gpkg_path = self.output_path.replace('.csv', '.gpkg')
            region_gdf.to_file(gpkg_path, driver='GPKG')
            print(f"地理数据已保存为: {gpkg_path}")
        except Exception as e:
            print(f"保存GeoPackage失败: {e}")
        
        return region_gdf

# 直接设置路径并运行
if __name__ == "__main__":
    # 直接设置路径
    model_dir = "E:/怀化市地质灾害攻关/全市易发性/result/传统算法/saved_models"
    input_path = "E:/怀化市地质灾害攻关/多尺度网格/csv/特征数据/100网格.csv"
    output_path = "E:/怀化市地质灾害攻关/全市易发性/result/传统算法/预测结果/传统算法预测结果100网格.csv"
    
    # 创建预测器并执行预测
    predictor = TraditionalStackingPredictor(model_dir, input_path, output_path)
    result = predictor.predict_risk()
    
    print("预测完成!")