# Landslide Susceptibility Modeling – Three‑Layer Stacking Ensemble with Mechanism‑Constrained Sampling

This project implements the complete workflow described in the paper:

> **A Novel Three‑Layer Stacking Ensemble with Geological‑Mechanism Constrained Sampling for Interpretable Landslide Susceptibility Modeling**  
> Shijun Xi, Jian Xiao, Lin Zen, Liping Teng, Tao Luo, Hongzuan Chen  
> *Submitted to Computers & Geosciences* (2025)

The repository contains two core scripts:

1. **`balancesample.py`**  
   Generates balanced positive‑negative sample sets based on geological mechanics and strict temporal matching. Negative samples are assigned dates sampled from the positive inventory, mimicking the physical scenario where landslides may or may not occur under the same rainfall event.

2. **`Stackingtechnology.py`**  
   Implements the three‑layer stacking ensemble, including base learners (Logistic, SVM, RandomForest, AdaBoost, XGBoost), an intermediate feature fusion layer, a meta‑learner, Bayesian hyperparameter optimisation, dynamic weighting, and SHAP interpretability (optional).

---

## 1. Requirements

- Python 3.11
- Required packages (install via conda or pip):
  ```
  numpy pandas matplotlib seaborn
  geopandas cartopy geopy
  scikit-learn xgboost
  scikit-optimize          # for Bayesian optimisation
  imbalanced-learn         # for SMOTE
  shap                     # optional, for interpretability
  tqdm
  ```

Installation example:
```bash
pip install numpy pandas matplotlib seaborn geopandas cartopy geopy scikit-learn xgboost scikit-optimize imbalanced-learn shap tqdm
```

> **Note**: `cartopy` may require conda on some systems: `conda install -c conda-forge cartopy`

---

## 2. Sample Balancing Script: `balancesample.py`

### 2.1 Description

This script selects negative samples from predefined susceptibility zones (classes 2 and 3) while respecting:

- **Spatial constraints**: minimum distance (`MIN_DISTANCE`, default 100 m) to any positive sample.
- **Temporal constraints**: negative samples are assigned dates drawn **with replacement** from the positive sample list, ensuring the month/year distribution exactly matches that of the positives. This simulates a “same rainfall event – different slope outcome” scenario.

Main steps:
- Read positive samples (CSV with columns: `id`, `lon`, `lat`, `日期` (date), `target=1`).
- Load Huaihua city administrative boundary and susceptibility zoning shapefile.
- Spatially filter positives: keep only those inside the boundary and within susceptibility classes 2 or 3.
- Analyse temporal distribution of positives (year, month, season, weekday).
- Randomly generate candidate negative points inside susceptibility zones 2 & 3.
- Apply minimum‑distance check against positives.
- For each accepted negative point, randomly sample a date from the positive date list.
- Output balanced datasets for multiple positive‑to‑negative ratios (controlled by `TARGET_RATIOS`) and generate comparison plots (spatial distribution + temporal histograms).

### 2.2 Configuration

Edit the parameters at the top of the script:

```python
BASE_DIR = r'E:/怀化市地质灾害攻关/样本数据/全市含降雨时间样本'   # output root
DATA_PATH = r'E:/怀化市地质灾害攻关/陈红专提供数据/全市正样本.csv'
BOUNDARY_PATH = 'E:/怀化市地质灾害攻关/约束数据/怀化市行政边界.shp'
VULNERABILITY_PATH = 'E:/怀化市地质灾害攻关/约束数据/易发性分区.shp'
MIN_DISTANCE = 100          # minimum distance between samples (meters)
RANDOM_STATE = 42
TARGET_RATIOS = [2,3,4,5,8,10]   # generates ratios 1:2, 1:3, ... 1:10
```

### 2.3 Input Data Format

- **Positive sample CSV** must contain:
  - `id` : unique identifier
  - `lon`, `lat` : longitude, latitude (WGS84)
  - `日期` : date of landslide occurrence (e.g., `2020-06-23`)
  - `target` : set to 1 (negative samples will be automatically assigned 0)

- **Susceptibility shapefile** must have a field `fenji` with values:
  - 1 = low susceptibility
  - 2 = moderate susceptibility
  - 3 = high susceptibility  
  Negative samples are only drawn from zones where `fenji` is 2 or 3.

### 2.4 Execution

Simply run:

```bash
python balancesample.py
```

For each ratio, the following are saved under `BASE_DIR/ratio_1_<ratio>/`:
- `balanced_ratio_1_<ratio>.csv` : the balanced dataset
- `comparison_ratio_1_<ratio>.png` : spatial distribution (with susceptibility background) and temporal distribution (month/year bar charts) comparison.

### 2.5 Notes

- The exact geodesic distance check (`geopy.distance.geodesic`) can be slow for many candidate points. If speed is an issue, set `MIN_DISTANCE = 0` to skip the check, or replace with a faster Euclidean approximation (keeping in mind that coordinates are in degrees).
- Temporal matching is achieved by sampling from the positive date list with replacement; therefore negative dates may repeat, but the overall monthly/yearly distribution closely mirrors that of the positives.
- The script automatically creates output directories. If not enough candidate points are found or the distance condition cannot be satisfied, warnings are issued and the script will attempt to supplement points (possibly violating the minimum distance) to meet the required count.

---

## 3. Model Training Script: `Stackingtechnology.py`

### 3.1 Description

Implements the three‑layer stacking ensemble. Workflow:

1. **Data preprocessing** (`DataProcessor` class):
   - Load balanced CSV, separate features and target.
   - Optional feature selection (`FEATURE_SELECTION = True`) using a Random Forest to keep only important features.
   - Standardisation (`StandardScaler`).
   - Optional imbalance handling (`BALANCE_METHOD = 'SMOTE'`).

2. **Base learner definition** (`BaseModel` class):
   - Supports five models: Logistic, SVM, RandomForest, AdaBoost, XGBoost.
   - Each model has predefined fixed parameters and a hyperparameter search space (`param_space`).
   - Bayesian optimisation (`BayesSearchCV`) tunes hyperparameters using `roc_auc` as the scoring metric.

3. **Three‑layer stacking architecture** (`EnsemblePipeline` class):
   - Split data into training, validation, and test sets (7:1:2).
   - **Base layer**: train five base models on the training set and generate out‑of‑fold (OOF) probabilities on the validation set.
   - **Intermediate layer**: concatenate the original validation features with the five OOF probability vectors, forming an augmented feature matrix (original dimensionality + 5).
   - **Meta‑layer**: train a meta‑model (default XGBoost) on this augmented matrix.
   - Evaluate on the test set, producing various metrics and generating ROC/PR curves and feature importance plots (if the meta‑model is tree‑based).

4. **Model persistence**: Save trained base models, meta‑model, scaler, and selected feature names as `.pkl` files for later deployment.

### 3.2 Configuration

The `Config` class at the top of the script holds all settings:

```python
class Config:
    TARGET_COL = 'target'
    TEST_SIZE = 0.3
    VAL_SIZE = 0.2
    RANDOM_STATE = 42

    BASE_MODELS = ['Logistic', 'SVM', 'RandomForest', 'AdaBoost', 'XGBoost']
    META_MODEL = 'XGBoost'          # can be 'Logistic' or 'XGBoost'

    N_ITER = 30                      # Bayesian optimisation iterations
    CV_FOLDS = 5
    SCORING = 'roc_auc'

    FEATURE_SELECTION = True
    BALANCE_METHOD = 'SMOTE'         # None or 'SMOTE'

    OUTPUT_ROOT = "C:/python_deepstudy/landslide/result/ratio_1_1/1km"
    FIGURE_SIZE = (10, 6)
```

### 3.3 Input Data Format

- A CSV file produced by the sample balancing script, containing all feature columns (e.g., elevation, slope, NDVI, ...) and a `target` column.
- The script currently expects the data at `DATA_PATH = "C:/python_deepstudy/landslide/result/ratio_1_1/平衡后的正负样本.csv"` – adjust this path accordingly.

### 3.4 Execution

```bash
python Stackingtechnology.py
```

After execution, the following are saved under `Config.OUTPUT_ROOT`:
- `figures/` : ROC curve, PR curve, feature importance plot (if applicable)
- `saved_models/` : five base models (`<name>_base.pkl`), meta‑model (`meta_model.pkl`), preprocessor info (`preprocessor.pkl`)
- `evaluation_report.json` : all evaluation metrics in JSON format

### 3.5 Notes

- Make sure `scikit-optimize` is installed for `BayesSearchCV`.
- Bayesian optimisation can be time‑consuming on large datasets; reduce `N_ITER` if needed.
- Feature selection changes the feature set. The preprocessor saved in `preprocessor.pkl` includes the scaler and the list of selected feature names, so that new data can be transformed identically during deployment.
- Currently only Logistic and XGBoost are supported as meta‑models. To use another classifier, extend `EnsemblePipeline._init_meta_model()` accordingly.

---

## 4. Complete Workflow Example

Assuming you have the positive sample CSV and shapefiles ready, the following commands run the full pipeline for a 1:5 ratio:

```bash
# Step 1: Generate balanced samples (make sure TARGET_RATIOS includes 5)
python balancesample_时空约束ok20251205.py

# Step 2: Edit Stackingtechnology.py to point to the generated file
# e.g. DATA_PATH = "E:/怀化市地质灾害攻关/样本数据/全市含降雨时间样本/ratio_1_5/balanced_ratio_1_5.csv"

# Step 3: Run model training
python Stackingtechnology.py

# Step 4: Inspect results
# - Open evaluation_report.json to see metrics
# - Look at figures/ for ROC, PR curves, etc.
```

---

## 5. Customisation & Extensions

### 5.1 Adding a New Base Model

Add a new entry to `BaseModel.model_config`:

```python
'NewModel': {
    'class': NewClassifier,
    'params': {'random_state': Config.RANDOM_STATE, ...},
    'param_space': {
        'param1': Real(0.1, 10.0),
        'param2': Integer(10, 100),
        ...
    }
}
```

Then include `'NewModel'` in `Config.BASE_MODELS`.

### 5.2 Modifying Sampling Strategy

To change the rules for negative sample selection (e.g., buffer sizes, susceptibility classes), edit the corresponding parts in `balancesample_...py`, such as the `filter_in_boundary` function or the candidate point generation loop.

### 5.3 Using a Different Meta‑Learner

Set `Config.META_MODEL` to the desired name and ensure the class is instantiated correctly in `EnsemblePipeline._init_meta_model()`.

---

## 6. Frequently Asked Questions

**Q: The geodesic distance check in `balancesample` is very slow. What can I do?**  
A: Temporarily set `MIN_DISTANCE = 0` to skip the check, or replace the check with a faster Euclidean approximation (but remember that coordinates are in degrees; you would need to convert to metres approximately).

**Q: Bayesian optimisation does not seem to find good parameters.**  
A: Increase `N_ITER` or narrow the search space based on prior experience. You can also run a few simple models manually to get a rough idea of reasonable parameter ranges.

**Q: How do I use the saved models to predict on a new area?**  
A: Load `preprocessor.pkl` and `meta_model.pkl` using `pickle`. Apply the same standardisation and feature selection to the new data, then call `meta_model.predict_proba()`.

**Q: How can I generate SHAP plots?**  
A: The script does not include SHAP by default, but you can easily add it after training:

```python
import shap
explainer = shap.TreeExplainer(pipeline.meta_model)   # works for tree‑based meta‑models
shap_values = explainer.shap_values(pipeline.meta_features_test)
shap.summary_plot(shap_values, pipeline.meta_features_test)
```

---

## 7. Citation & Contact

If you use this code in your research, please cite our paper:

```bibtex
@article{xi2025three,
  title={A Novel Three-Layer Stacking Ensemble with Geological-Mechanism Constrained Sampling for Interpretable Landslide Susceptibility Modeling},
  author={Xi, Shijun and Xiao, Jian and Zen, Lin and Teng, Liping and Luo, Tao and Chen, Hongzuan},
  journal={Computers \& Geosciences},
  year={2025},
  note={Under review}
}
```

The code is available at: [https://github.com/H432830/Susceptibility-mapping](https://github.com/H432830/Susceptibility-mapping)  
For questions or collaborations, contact the corresponding author: **Shijun Xi** (📧 m19984468477@163.com)

---
