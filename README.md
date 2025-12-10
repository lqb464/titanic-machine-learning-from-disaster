# Titanic - Machine Learning from Disaster

A Kaggle classification project that predicts which passengers survived the sinking of the Titanic using passenger information such as age, sex, ticket class, fare, and engineered features.

Competition link: https://www.kaggle.com/competitions/titanic  

> Lưu ý: EDA và toàn bộ phân tích trực quan nằm trong Notebook. Pipeline xử lý dữ liệu, tạo feature, huấn luyện mô hình và tạo submission cũng nằm trong Notebook vì đây là lần đầu làm một project nên chưa quen với việc chia nhỏ.

---

## Repository Structure

```text
titanic-machine-learning-from-disaster/
├── dataset/         # train.csv, test.csv
├── notebook/        # Jupyter Notebook cho EDA
│   └── titanic-machine-learning-from-disaster.ipynb
├── document/        # Tài liệu báo cáo nếu có
├── output/          # submission.csv
├── requirements.txt
└── README.md
````

---

## Problem Description

Mục tiêu của bài toán:

> Dự đoán khả năng sống sót của hành khách dựa trên thông tin cá nhân như tuổi, giới tính, hạng vé và nhiều đặc trưng được trích xuất thêm.

Dữ liệu bao gồm:

* `train.csv`: có cột mục tiêu `Survived`
* `test.csv`: không có `Survived`, dùng để dự đoán nộp Kaggle

Quy trình tổng quát:

1. EDA trong notebook để hiểu dữ liệu
2. Feature engineering như Title, FamilySize, Cabin flag, FarePerPassenger
3. Xây dựng preprocessing pipeline
4. Train baseline models và thử PCA
5. Feature importance và feature selection
6. Chọn mô hình tốt nhất
7. Sinh file submission

---

## Approach Summary

### 1. Data validation & EDA (trong notebook)

Bao gồm:

* Missing values
* Data types
* Correlation với target
* Phân tích phân phối Age, Fare, Pclass, Sex, Embarked
* Survival rate theo từng nhóm

### 2. Feature Engineering

Một số feature quan trọng:

* **Title** từ Name
* **FamilySize**, **IsAlone**, **FamilySize_Bucketized**
* **HasCabin**
* **FarePerPassenger**, **LogFare**

Các feature này được chuyển thành module trong `src/features.py`.

### 3. Preprocessing Pipeline

Dùng:

* `SimpleImputer`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* `Pipeline`

Mã trong `src/preprocessing.py`.

### 4. Baseline Models

Thử nghiệm các mô hình:

* Logistic Regression
* Random Forest
* SVC
* XGBoost
* KNN

Kèm bản PCA để so sánh.

### 5. Feature Selection & Scenarios

* Rút trích feature importance từ RF và XGB
* So sánh tập feature đầy đủ vs tập rút gọn
* Đánh giá hai scenario: giữ tất cả feature hoặc giữ 5 feature mạnh nhất

### 6. Final Model

Mô hình chọn cuối cùng:

* **Gaussian Process với hyperparameter tuning**
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
source .venv/bin/activate     # hoặc .venv\Scripts\activate trên Windows

pip install -r requirements.txt
```

---

## Running the Notebook (EDA)

```bash
jupyter notebook notebook/titanic-machine-learning-from-disaster.ipynb
```

Notebook chứa toàn bộ nội dung:

* Data validation
* EDA
* Biểu đồ trực quan
* Phân tích survival rate
* Giải thích feature engineering
* So sánh mô hình