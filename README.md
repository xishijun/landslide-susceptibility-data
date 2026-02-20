# Landslide Susceptibility Modeling – Three‑Layer Stacking Ensemble with Mechanism‑Constrained Sampling

This project implements the complete workflow described in the paper:

> **A Novel Three‑Layer Stacking Ensemble with Geological‑Mechanism Constrained Sampling for Interpretable Landslide Susceptibility Modeling**  
> Shijun Xi, Jian Xiao, Lin Zen, Liping Teng, Tao Luo, Hongzuan Chen  


The repository contains a full implementation of three stacking ensemble algorithms, a unified data preprocessor, model evaluation and comparison tools, uncertainty analysis modules, and prediction utilities.

## 1. Project Overview

To address the common limitations in landslide susceptibility modelling—such as physically uninformative sample selection, shallow feature fusion, and static model integration—this study proposes a three‑layer stacking framework. The codebase includes:

- **Unified Data Processor** `UnifiedDataProcessor`: Handles feature selection, multicollinearity diagnostics (VIF/Tolerance), standardisation, and class imbalance (SMOTE).
- **Three Stacking Algorithms**:
  - `Stacking.py`: Traditional stacking (based on `sklearn.ensemble.StackingClassifier`).
  - `Stackingtechnology.py`: Two‑layer stacking (base model outputs + original features → meta‑model).
  - `three_layer_Stacking.py`: Three‑layer stacking (base models → intermediate models → meta‑model, with weighted fusion).
- **Unified Model Evaluator** `PaperReadyEvaluator`: Generates publication‑quality figures (ROC, PR, confusion matrix, calibration curve, radar chart, etc.) and comparison reports.
- **Uncertainty Analysis** `uncertainty_analysis.py`: Quantifies prediction uncertainty using Monte Carlo simulation, bootstrap resampling, and SHAP.
- **Predictors**: `TraditionalStackingPredictor`, `TwoLayerStackingPredictor`, `ThreeLayerStackingPredictor` – load trained models and apply them to new areas.

## 2. File Structure

```
├── data_processor.py                # Unified data preprocessing
├── model_evaluator.py                # Unified evaluation and plotting
├── run_all_algorithms.py             # Master script (runs all three algorithms and compares)
├── Stacking.py                        # Traditional stacking algorithm
├── Stackingtechnology.py              # Two‑layer stacking algorithm
├── three_layer_Stacking.py            # Three‑layer stacking algorithm
├── TraditionalStackingPredictor.py    # Predictor for traditional algorithm
├── TwoLayerStackingPredictor.py       # Predictor for two‑layer algorithm
├── ThreeLayerStackingPredictor.py     # Predictor for three‑layer algorithm
├── uncertainty_analysis.py            # Uncertainty analysis
└── README.md                          # This file
```

## 3. Requirements

- Python 3.11
- Required packages (install with conda or pip):

```bash
pip install numpy pandas matplotlib seaborn geopandas scikit-learn xgboost scikit-optimize imbalanced-learn shap tqdm statsmodels
```

If you need to save GeoPackage files, ensure that `fiona` and `geopandas` are correctly installed (they usually come with the geopandas installation).

## 4. Data Preparation

Input data should be a CSV file containing the following columns:
- `lon`, `lat`: Longitude and latitude (WGS84).
- Environmental factor columns (e.g., `DEM`, `slope`, `NDVI`, etc.; the exact names must match those defined in `self.all_features` in the code).
- `target`: The label column (1 = landslide, 0 = non‑landslide).

You can modify the data path in the `Config` class of each algorithm script, or directly set `DATA_PATH` in `run_all_algorithms.py`.

## 5. Configuration Parameters

Each algorithm script contains a `Config` class at the top, where all adjustable parameters are centralised. The main parameters are listed below:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TARGET_COL` | Name of the target column | `'target'` |
| `TEST_SIZE` | Proportion of test set | 0.2 ~ 0.3 |
| `VAL_SIZE` | Proportion of validation set (two‑/three‑layer) | 0.2 |
| `RANDOM_STATE` | Random seed | 42 |
| `BASE_MODELS` / `LAYER1_MODELS` | List of base learners | See each Config |
| `META_MODEL` / `META_MODEL_TYPE` | Meta‑learner type | `'XGBoost'` |
| `N_ITER` | Number of Bayesian optimisation iterations | 15–30 |
| `CV_FOLDS` | Number of cross‑validation folds | 5 |
| `SCORING` | Optimisation metric | `'roc_auc'` |
| `FEATURE_SELECTION` | Whether to perform feature selection | `True` |
| `BALANCE_METHOD` | Method for handling class imbalance | `'SMOTE'` |
| `OUTPUT_ROOT` | Root directory for results | Separate for each algorithm |
| `USE_ORIGINAL_FEATURES` | Whether to include original features in meta‑features (three‑layer) | `True` |
| `WEIGHTED_STACKING` | Whether to use weighted fusion (three‑layer) | `True` |

## 6. Quick Start

### 6.1 Run All Three Algorithms and Generate Comparison Report

Simply execute the master script:

```bash
python run_all_algorithms.py
```

This script will:
- Use `UnifiedDataProcessor` to load and preprocess the data (generating VIF/tolerance analysis, feature importance, and correlation heatmaps).
- Run traditional, two‑layer, and three‑layer stacking algorithms sequentially, saving models and evaluation results in their respective output directories.
- Finally, call `PaperReadyEvaluator` to produce comparison plots (ROC curves, radar chart, bar chart, etc.) and a comprehensive report.

### 6.2 Run a Single Algorithm

If you only need to run one algorithm, execute the corresponding script directly, for example:

```bash
python three_layer_Stacking.py
```

Each algorithm script will run independently and output results to the directory specified by `OUTPUT_ROOT`.

### 6.3 Uncertainty Analysis

After training, you can run uncertainty analysis:

```bash
python uncertainty_analysis.py
```

This script loads the trained three‑layer model, performs Monte Carlo simulation and SHAP analysis on a given regional dataset, and generates uncertainty maps, statistical plots, and a comprehensive report.

**Note:** Modify the paths for `model_dir`, `data_path`, and `output_dir` inside the script before running.

### 6.4 Predict on a New Area

Use the corresponding predictor script, for example:

```bash
python ThreeLayerStackingPredictor.py
```

The predictor automatically loads the models, scaler, and feature column information from the `saved_models` directory, computes probability predictions for every point in the input CSV, and saves the results (with `lon`, `lat`, `probability`) as a new CSV file.

## 7. Output Description

### 7.1 Per‑Algorithm Output Directory

- `figures/`: ROC curve, PR curve, confusion matrix, calibration curve, probability distribution, etc.
- `saved_models/`: Base models, meta‑model, scaler, feature column information.
- `evaluation_report.json`: All evaluation metrics in JSON format.
- `base_models_performance.csv`: Cross‑validation performance of base models.

### 7.2 Algorithm Comparison Output

Under `OUTPUT_ROOT/comparison_figures/`:
- `roc_curve_comparison.png`: ROC curves of the three algorithms.
- `pr_curve_comparison.png`: Precision‑Recall curves.
- `performance_radar.png`: Radar chart of key metrics.
- `metrics_bar_chart.png`: Bar chart comparison.
- `comprehensive_comparison.csv` and `.tex`: Comprehensive metrics table.

### 7.3 Uncertainty Analysis Output

- `shap_plots/`: SHAP summary plots, dependence plots, waterfall plots.
- `uncertainty_maps/`: Spatial distribution of various uncertainty metrics.
- `statistical_plots/`: Histograms, correlation heatmaps of uncertainty metrics.
- `comprehensive_report/`: Uncertainty statistics tables and classification reports.

## 8. Customisation and Extensions

### 8.1 Adding a New Base Model

Add a new entry to `BaseModel.model_config` following the format of existing models. Then include the model name in the corresponding `Config.BASE_MODELS` or `LAYER1_MODELS` list.

### 8.2 Modifying Feature Selection

`UnifiedDataProcessor` currently uses a combination of random‑forest‑based selection and F‑test selection. You can change the feature selection method by editing the `_feature_selection` method (e.g., using VIF‑based filtering, PCA, etc.).

### 8.3 Adjusting Bayesian Optimisation

Modify `N_ITER` and `CV_FOLDS` in `Config` to control the trade‑off between optimisation accuracy and runtime. For large datasets, consider reducing the number of iterations.

## 9. Frequently Asked Questions

**Q: I get a "module not found" error when running `run_all_algorithms.py`.**  
A: Ensure all required packages are installed. Pay special attention to `scikit-optimize` and `imbalanced-learn`.

**Q: Data preprocessing fails with "missing required columns".**  
A: Check that your input CSV contains all the expected feature columns and that their names match those in `self.all_features` in the code. You can exclude non‑feature columns (like longitude/latitude) via the `exclude_columns` parameter in `UnifiedDataProcessor`.

**Q: How do I change the sample ratio or handle class imbalance differently?**  
A: Set `BALANCE_METHOD = 'SMOTE'` in `Config` to automatically oversample the minority class. If you need a specific ratio, you should prepare a balanced sample set beforehand (e.g., using the separate `balancesample` script mentioned in the paper).

**Q: Prediction fails due to feature count mismatch.**  
A: The predictor uses `feature_columns.pkl` and `scaler.pkl` saved during training. Ensure your input data contains exactly those columns and that they are in the correct order.

## 10. Citation and Contact


If you use this code in your research:

The code is available at: [https://github.com/xishijun/landslide-susceptibility-data/tree/main))] 
For questions or collaborations, contact the corresponding author: **Shijun Xi** (📧 m19984468477@163.com)

---
