import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import os
os.makedirs("figures", exist_ok=True)

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_val = np.load("data/X_val.npy")
y_val = np.load("data/y_val.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)

lr_val_pred = lr_model.predict(X_val)
lr_val_rmse = np.sqrt(mean_squared_error(y_val, lr_val_pred))
print(f"Linear Regression Validation RMSE: {lr_val_rmse}")

rf_val_pred = rf_model.predict(X_val)
rf_val_rmse = np.sqrt(mean_squared_error(y_val, rf_val_pred))
print(f"Random Forest Validation RMSE: {rf_val_rmse}")
lr_test_pred = lr_model.predict(X_test)
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
print(f"Linear Regression Test RMSE: {lr_test_rmse}")

rf_test_pred = rf_model.predict(X_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
print(f"Random Forest Test RMSE: {rf_test_rmse}")

importances = rf_model.feature_importances_
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.savefig("figures/feature_importances.png")
plt.close()
lr_residuals = y_test - lr_test_pred
plt.figure(figsize=(8, 6))
plt.scatter(lr_test_pred, lr_residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title("Linear Regression Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.savefig("figures/residual_plot.png")
plt.close()
