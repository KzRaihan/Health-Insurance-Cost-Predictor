# Health Insurance Cost Predictor

**Predicting individual medical insurance charges using machine learning**

![ML Project](https://upload.wikimedia.org/wikipedia/commons/4/4c/ML_logo.png)

### 📋 Project Overview

This project builds and compares several regression models to predict **personal medical insurance charges** based on demographic and health-related features.

The work demonstrates a complete end-to-end machine learning workflow:

- Exploratory Data Analysis (EDA)
- Data preprocessing & feature engineering
- Model selection & comparison
- Hyperparameter tuning
- Performance evaluation & interpretation

Main goal: Understand which factors most strongly influence insurance costs and build the most accurate possible prediction model.

### 🎯 Dataset

**Kaggle – Medical Cost Personal Datasets**
https://www.kaggle.com/datasets/mirichoi0218/insurance

**Features:**

- `age` — age of primary beneficiary (18–64)
- `sex` — gender (male/female)
- `bmi` — body mass index (kg/m²)
- `children` — number of children/dependents covered
- `smoker` — smoking status (yes/no)
- `region` — residential area in the US (northeast, northwest, southeast, southwest)
- `charges` — individual medical costs billed by health insurance (target)

**Size:** 1,338 records

### 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Core libraries**:
  - pandas · numpy
  - scikit-learn
  - xgboost / lightgbm (optional)
  - ydata-profiling
  - matplotlib · seaborn
- **Model selection & tuning**: GridSearchCV / RandomizedSearchCV / Optuna (recommended)
- **Environment management**: conda / venv

### 📊 Models Implemented & Compared

The following regression models were trained and evaluated on the test set:

| Rank | Model                    | R² Score        | RMSE              | MAE               | Comment                               |
| ---- | ------------------------ | ---------------- | ----------------- | ----------------- | ------------------------------------- |
| 1    | Gradient Boosting        | **0.9007** | **4271.89** | **2535.06** | Best single model performance         |
| 2    | Stacking Ensemble        | 0.8977           | 4336.24           | **2500.24** | Excellent – best MAE                 |
| 3    | Random Forest            | 0.8809           | 4678.88           | 2586.37           | Very strong & robust baseline         |
| 4    | Voting Ensemble          | 0.8384           | 5449.79           | 3241.15           | Decent, but outperformed by others    |
| 5    | Linear Regression        | 0.8069           | 5956.34           | 4177.05           | Good interpretable baseline           |
| 6    | Support Vector Regressor | 0.3774           | 10695.79          | 5180.22           | Poor performance – not suitable here |

**Key observations:**

- **Gradient Boosting** achieves the highest R² and very competitive RMSE
- **Stacking Ensemble** gives the lowest MAE (best absolute error control)
- Tree-based ensemble methods clearly dominate linear & SVR approaches
- Final choice between **Gradient Boosting** and **Stacking** depends on whether you prioritize R²/RMSE or MAE

Best overall model : **Gradient Boosting Regressor**

### 🗂️ Project Structure

Health-Insurance-Cost-Predictor/
│
├── app.py
├── model/
│   └── insurance_gb_pipeline.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Comparison.ipynb
│   └── 03_Hyperparameter_Tuning.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore

### 🧪 How to Run Locally

1️⃣ Clone the repository

```bash
git clone https://github.com/KzRaihan/Health-Insurance-Cost-Predictor.git
cd Health-Insurance-Cost-Predictor
```

2️⃣ Create a Virtual Environment

```bash
 conda create -n mlProject python=3.11 -y
```

3️⃣ Activate virtual environment

```bash
 conda activate mlProject
```

4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

5️⃣ Run the Gradio App

```bash
python app.py
```

## 📌 Future Improvements

- **Implement CI/CD & MLOps Practices** —:
  - Automated testing
  - Model training & evaluation on push
  - Model registry (MLflow + DVC)

### 👨‍💻 Author

**Md Kamruzzaman Raihan**

BSc in Computer Science & Engineering

Focus: Machine Learning, Deep Learning, Applied AI

### 📄 License

This project is licensed under the **MIT License** and it's open-source and free to use for learning purposes.
