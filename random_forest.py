#Random Forest
import kaggle
import pandas as pd 
import zipfile
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_train_rf = X_train.sample(n=500000, random_state=42)
y_train_rf = y_train.loc[X_train_rf.index]
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), ["state", "city", "zip_code"])
], remainder='passthrough')

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        max_features='sqrt',
        verbose=1))
])

print("\nTraining Random Forest model...")
rf_model.fit(X_train_rf, y_train_rf)

y_pred_log_rf = rf_model.predict(X_test)
y_pred_rf = np.expm1(y_pred_log_rf)
print("\nEvaluating Random Forest model...")
print("MAE (RF):", mean_absolute_error(y_true, y_pred_rf))
print("R2 score (RF):", r2_score(y_true, y_pred_rf))
print("RMSE (RF):", np.sqrt(np.mean((y_true - y_pred_rf) ** 2)))

# 1. Predicted vs Actual scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred_rf, alpha=0.3, s=10)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Random Forest: Predicted vs Actual House Prices')
plt.legend()
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Residuals plot (errors)
residuals = y_true - y_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price ($)')
plt.ylabel('Residuals ($)')
plt.title('Random Forest: Residual Plot')
plt.ticklabel_format(style='plain', axis='both')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Error distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error ($)')
plt.ylabel('Frequency')
plt.title('Random Forest: Distribution of Prediction Errors')
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.ticklabel_format(style='plain', axis='x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Feature importance (top 15 features)
feature_names = rf_model.named_steps['preprocessor'].get_feature_names_out()
importances = rf_model.named_steps['regressor'].feature_importances_

# Get top 15
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Random Forest: Top 15 Most Important Features')
plt.tight_layout()
plt.show()