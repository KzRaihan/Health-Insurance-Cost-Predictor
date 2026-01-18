### === import necessary libraries === ###
import numpy as np
import pandas as pd
import pickle


## Sklearn Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Model 
from sklearn.ensemble import GradientBoostingRegressor

#### Evaluation Metrices
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================
# -  Load dataset
# =====================
df = pd.read_csv("insurance.csv")

# =====================
# -  keeps the first occurance(Duplicate)
# =====================
df = df.drop_duplicates()

# =====================
# -  Features matrix and Target Variable
# =====================

X = df.drop('charges', axis = 1)
y = df['charges']

# =====================
# - Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =====================
# - Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================
# - Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])


# =====================
#  - GradientBoostingRegressor
# =====================
reg_gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate = 0.08,
    max_depth = 3,
    min_samples_split = 8,
    subsample = 0.8,
    random_state=42
    )


# =====================
# - Full Pipeline
# =====================
gb_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', reg_gb)
])

# =====================
# - fit the model
# =====================
gb_pipe.fit(X_train, y_train)


# =====================
# - Evaluation
# =====================
y_pred = gb_pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")



# =====================
# - Save model 
# =====================
with open("insurance_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipe, f)

print("✅ pipeline saved as insurance_gb_pipeline.pkl")
