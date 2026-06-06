# Machine Learning Projects: Kaggle Competition AMEX

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/amex-default-prediction)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)

## 📌 Project Overview & Purpose
The objective of this project is to build a high-performance predictive model that determines the probability that a customer will default on their credit card balance. Operating at the scale of American Express (the world's largest payment card issuer), even fractional improvements in prediction accuracy translate to billions of dollars in mitigated risk. 

From a data science perspective, this project serves as a robust application of **large-scale time-series aggregation**, handling **severe class imbalance**, and deploying **gradient-boosted decision trees (GBDTs)** to solve complex tabular classification problems.

---

## 📊 Dataset Profile & Size
* **Source:** [Kaggle AMEX Default Prediction Competition](https://www.kaggle.com/competitions/amex-default-prediction)
* **Dataset Size:** ~50 GB (Raw CSV format)
* **Data Structure:** Large-scale time-series data with multiple monthly statements per customer (`customer_ID`). 
* **Features:** 188 anonymized and normalized variables split into five behavioral categories:
  * `D_*`: Delinquency variables
  * `S_*`: Spend variables
  * `P_*`: Payment variables
  * `B_*`: Balance variables
  * `R_*`: Risk variables
* **Target:** Binary variable (`0` = No default, `1` = Default within a 120-day window after the latest statement).
* **Class Imbalance:** The negative class (`0`) was subsampled to 5% in the training set, meaning a **20x weighting** must be applied during metric scoring.

---

## 🛠️ My End-to-End Workflow & Contributions
> 💡 **Note on Contributions:** I designed and executed the entire data engineering pipeline, feature engineering strategy, model training, and evaluation workflows detailed below.

### 1. Data-Cleaning & Memory Optimization Workflow
Handling a 50 GB dataset on standard compute required strict memory management protocols:
* **Downcasting Data Types:** Converted 64-bit floats and integers to `float32` and `int8`/`int16` where possible, reducing the memory footprint by over 60%.
* **Parquet Conversion:** Converted the raw `.csv` files into compressed `.parquet` files to drastically speed up disk I/O operations.
* **Missing Value Imputation:** Categorical missing entries were treated as a distinct class (`"Unknown"`), while numerical missing values were handled natively by the tree-based algorithms.

### 2. Feature-Engineering Approach
Beyond using raw behavioral data, I engineered two primary categories of meta-features:
* **Statistical Distribution Triggers (Within-Category Deviation):**
  * Grouped features within the **Delinquency (`D_*`)** and **Balance (`B_*`)** blocks to calculate a dynamic boundary based on standard deviation and mean offsets ($\mu \pm 0.4\sigma$ and $\mu + 1\sigma$).
  * Formulated index counts tracking how many individual variables for a single customer crossed above or below those custom thresholds (`count_D_up`, `count_D_down`, etc.).
* **Custom Interaction Features (`combine_label`):**
  * Designed and programmed an automated algorithmic class (`combine_label`) that normalizes and projects interaction pairs between a dominant column (e.g., `S_22`) and all other structural numerical elements into a discrete matrix map using 100 specific bins.
    
### 3. Cross-Validation Method
To ensure robust local evaluation and prevent data leakage:
* **Stratified K-Fold (5 Folds):** Implemented a 5-fold cross-validation scheme stratified by the target label to preserve the highly skewed default ratio across all folds.
* **Out-of-Fold (OOF) Predictions:** Generated OOF predictions to compute a reliable local evaluation score tracking closely with the Kaggle leaderboard.

### 4. Models Evaluated
I evaluated three state-of-the-art GBDT architectures known for handling tabular data efficiently:
1. **LightGBM:** Chosen for its rapid training speed and native support for missing values.
2. **XGBoost:** Deployed with histogram-based tree growing (`tree_method='hist'`) for optimized memory performance.
3. **CatBoost:** Leveraged specifically to evaluate its proprietary symmetric tree structures on categorical interactions.

---

## 🏆 Results & Interpretation

### Final Score
The models were evaluated using the unique AMEX competition metric (the mean of the Normalized Gini Coefficient and the default capture rate at 4%).

| Model | 5-Fold OOF CV Score | Public Leaderboard | Status |
| :--- | :---: | :---: | :---: |
| Baseline LightGBM | 0.782 | 0.781 | Completed |
| **Optimized XGBoost** | **0.794** | **0.793** | **Best Model** |
| CatBoost | 0.789 | 0.788 | Completed |

### Analysis & Interpretation
* **Feature Importance:** Aggregated features derived from **Payment (`P_*`)** and **Delinquency (`D_*`)** indicators consistently ranked as the highest predictors of credit default. In particular, the *latest* recorded payment behavior (`P_2_last`) was the single most dominant feature.
* **Model Insights:** XGBoost slightly outperformed LightGBM after extensive hyperparameter tuning (learning rate scheduling and regularization adjustments), successfully capturing non-linear risk interactions without overfitting.

---

## 📈 Visualizations
*(Uncomment and update the file paths below once you save your plot images to the repository)*

---

## 💻 Instructions for Running the Code

### 1. Prerequisites & Required Python Packages
Ensure you have Python 3.8+ installed. Install the required dependencies using:

```bash
pip install numpy pandas lightgbm xgboost catboost scikit-learn pyarrow
