import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)

from sklearn.metrics import r2_score, mean_absolute_error

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("data/cardekho.csv")

print("Dataset Shape:", df.shape)

# ======================================
# Drop Unnecessary Columns
# ======================================

drop_cols = ["Unnamed: 0"]

for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ======================================
# Features and Target
# ======================================

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# Numerical and Categorical Columns
# ======================================

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# ======================================
# Preprocessing
# ======================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ======================================
# Base Regressors
# ======================================

base_models = [

    (
        "lr",
        LinearRegression()
    ),

    (
        "rf",
        RandomForestRegressor(
            n_estimators=20,
            max_depth=10,
            random_state=42,
            n_jobs=1
        )
    ),

    (
        "gb",
        GradientBoostingRegressor(
            n_estimators=50,
            random_state=42
        )
    )
]

# ======================================
# Meta Regressor
# ======================================

meta_model = LinearRegression()

# ======================================
# Stacking Regressor
# ======================================

stack_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,
    n_jobs=1
)

# ======================================
# Final Pipeline
# ======================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", stack_regressor)
])

# ======================================
# Train Model
# ======================================

print("Training Stacking Regressor...")

model.fit(X_train, y_train)

# ======================================
# Predictions
# ======================================

y_pred = model.predict(X_test)

# ======================================
# Evaluation
# ======================================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nSTACKING REGRESSION RESULTS")
print("=" * 40)
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")

# ======================================
# Individual Model Comparison
# ======================================

lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",
     RandomForestRegressor(
         n_estimators=20,
         max_depth=10,
         random_state=42,
         n_jobs=1
     ))
])

gb_model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",
     GradientBoostingRegressor(
         n_estimators=50,
         random_state=42
     ))
])

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

print("\nMODEL COMPARISON")
print("=" * 40)

print("Linear Regression R2  :", round(r2_score(y_test, lr_pred), 4))
print("Random Forest R2      :", round(r2_score(y_test, rf_pred), 4))
print("Gradient Boosting R2  :", round(r2_score(y_test, gb_pred), 4))
print("Stacking Regressor R2 :", round(r2, 4))

# ======================================
# Save Compressed Model
# ======================================

joblib.dump(
    model,
    "models/stacking_regressor.pkl",
    compress=3
)

print("\nModel Saved Successfully!")
print("Location : models/stacking_regressor.pkl")
