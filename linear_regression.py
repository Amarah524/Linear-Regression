import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["Price"] = housing.target
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
x = df.drop("Price", axis=1)
y= df["Price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("\nTraining Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_pred_linear = linear_model.predict(x_test)
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_mae = mean_absolute_error(y_test, y_pred_linear)
print("\nLinear Regression Performance:")
print("MSE:", linear_mse)
print("MAE:", linear_mae)
residuals = y_test - y_pred_linear
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_linear, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Linear Regression")
plt.show()
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train, y_train)
y_pred_ridge = ridge_model.predict(x_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
print("\nRidge Regression Performance:")
print("MSE:", ridge_mse)
print("MAE:", ridge_mae)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
y_pred_lasso = lasso_model.predict(x_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
print("\nLasso Regression Performance:")
print("MSE:", lasso_mse)
print("MAE:", lasso_mae)

metrics_df = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression", "Lasso Regression"],
    "MSE": [linear_mse, ridge_mse, lasso_mse],
    "MAE": [linear_mae, ridge_mae, lasso_mae]
})
print("\nModel Performance Comparison:")
print(metrics_df)
coef_df = pd.DataFrame({
    "Feature": x.columns,
    "Linear": linear_model.coef_,
    "Ridge": ridge_model.coef_,
    "Lasso": lasso_model.coef_
})
print("\nModel Coefficient:")
print(coef_df)
coef_df.set_index("Feature").plot(kind="bar", figsize=(12,6))
plt.title("Coefficient Comparison")
plt.ylabel("Coefficient Value")
plt.show()
