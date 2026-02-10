# Titanic - Machine Learning from Disaster

A Kaggle classification project that predicts which passengers survived the sinking of the Titanic using passenger information such as age, sex, ticket class, fare, and engineered features.

Competition link: https://www.kaggle.com/competitions/titanic  

> Note: All EDA and visual analysis are contained in the Notebook. The data processing pipeline, feature engineering, model training, and submission generation are also implemented in the Notebook because this is my first full project and I am not yet fully comfortable splitting components into separate modules.

---

## Repository Structure

```text
titanic-machine-learning-from-disaster/
├── dataset/         # train.csv, test.csv
├── notebook/        # Jupyter Notebook for EDA
│   └── titanic-machine-learning-from-disaster.ipynb
├── document/        # Report documents if available
├── output/          # submission.csv
├── requirements.txt
└── README.md
````

---

## Problem Description

Problem objective:

> Predict passenger survival probability based on personal information such as age, gender, ticket class, and additional engineered features.

The dataset includes:

* `train.csv`: contains the target column `Survived`
* `test.csv`: does not contain `Survived`, used to generate Kaggle submissions

General workflow:

1. Perform EDA in the notebook to understand the data
2. Feature engineering such as Title, FamilySize, Cabin flag, FarePerPassenger
3. Build preprocessing pipeline
4. Train baseline models and experiment with PCA
5. Feature importance and feature selection
6. Select the best model
7. Generate submission file

---

## Approach Summary

### 1. Data validation & EDA (in notebook)

Includes:

* Missing values analysis
* Data types checking
* Correlation with target
* Distribution analysis of Age, Fare, Pclass, Sex, Embarked
* Survival rate by group

### 2. Feature Engineering

Some important features:

* **Title** extracted from Name
* **FamilySize**, **IsAlone**, **FamilySize_Bucketized**
* **HasCabin**
* **FarePerPassenger**, **LogFare**

These features are implemented as modules in `src/features.py`.

### 3. Preprocessing Pipeline

Using:

* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* `Pipeline`

Code located in `src/preprocessing.py`.

### 4. Baseline Models

Tested models:

* Logistic Regression
* Random Forest
* SVC
* XGBoost
* KNN

Including PCA variants for comparison.

### 5. Feature Selection & Scenarios

* Extract feature importance from RF and XGB
* Compare full feature set vs reduced feature set
* Evaluate two scenarios: keep all features or keep the top 5 most important features

### 6. Final Model

Final selected model:

* **Gaussian Process with hyperparameter tuning**
* Accuracy ~ **83.29%**
* F1-score ~ **77.30%**

---

## Technologies Used

* Python
* NumPy, Pandas
* Matplotlib, Seaborn, Plotly
* Scikit-learn
* XGBoost
* Statsmodels
* SciPy

---

## Installation

```bash
git clone <https://github.com/LQB464/titanic-machine-learning-from-disaster>
cd titanic-machine-learning-from-disaster

python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## Running the Notebook (EDA)

```bash
jupyter notebook notebook/titanic-machine-learning-from-disaster.ipynb
```

The notebook contains:

* Data validation
* EDA
* Visualizations
* Survival rate analysis
* Feature engineering explanation
* Model comparison
