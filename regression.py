import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from math import sqrt
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('bloodpressure-23.csv')
X = data[['SERUM-CHOL']]
y = data['SYSTOLIC']

# Cholesterol is ~200, so cholesterol^12 is ~1e27 and the fit falls apart
# numerically. Scale it first, inside the pipeline so each CV fold is
# scaled on its own training data only.
def polynomial_model(degree):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree), LinearRegression())

# Function to calculate RMSE
def calculate_rmse(X, y, degree):
    model = polynomial_model(degree)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    mse_values = -scores
    mean_mse = np.mean(mse_values)
    rmse = sqrt(mean_mse)
    return rmse

# Find the best degree
rmse_values = [calculate_rmse(X, y, degree) for degree in range(1, 15)]
best_degree = np.argmin(rmse_values) + 1

# Print RMSE values
print("Polynomial Regression:\n")
for degree, rmse in enumerate(rmse_values, start=1):
    print(f"Degree {degree}: RMSE = {rmse}")

# Plot RMSE vs. Polynomial Degree
plt.figure(figsize=(10, 6))
plt.plot(range(1, 15), rmse_values, marker='o', linestyle='-')
plt.title('Cross Validation RMSE vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean RMSE')
plt.grid(True)
plt.xticks(range(1, 15))
plt.yscale('log')  # high degrees overfit by orders of magnitude
plt.savefig('rmse_by_degree.png', dpi=120, bbox_inches='tight')
plt.show()

# Fit the best model
model = polynomial_model(best_degree)
model.fit(X, y)

# Print the coefficients (on the scaled cholesterol)
intercept = model[-1].intercept_
coefficients = model[-1].coef_
print(f"\nBest Degree: {best_degree}")
print(f"Intercept: {intercept}")
print("Coefficients:")
for i, coef in enumerate(coefficients):
    print(f"  Coefficient {i}: {coef}")

# Perform multiple linear regression on all features
X = data[['AGE', 'ED-LEVEL', 'SMOKING STATUS', 'EXERCISE', 'WEIGHT', 'SERUM-CHOL', 'IQ', 'SODIUM']]
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_values = np.sqrt(-scores)
mean_rmse = np.mean(rmse_values)

model.fit(X, y)
coefficients = model.coef_
intercept = model.intercept_

# Print coefficients and RMSE
print("\nCoefficients (Multiple Linear Regression):")
for feature, coef in zip(X.columns, coefficients):
    print(f"  {feature}: {coef}")
print(f"Intercept(Multiple Linear Regression): {intercept}")
print(f"Mean RMSE(Multiple Linear Regression): {mean_rmse}")

# Ridge Regression
alpha = 0.1
model_ridge = Ridge(alpha=alpha)
scores_ridge = cross_val_score(model_ridge, X, y, cv=10, scoring='neg_mean_squared_error')
rmse_values_ridge = np.sqrt(-scores_ridge)
mean_rmse_ridge = np.mean(rmse_values_ridge)

model_ridge.fit(X, y)
coefficients_ridge = model_ridge.coef_
intercept_ridge = model_ridge.intercept_

# Print coefficients and RMSE for Ridge Regression
print("\nCoefficients (Ridge Regression):")
for feature, coef in zip(X.columns, coefficients_ridge):
    print(f"  {feature}: {coef}")
print(f"Intercept (Ridge Regression): {intercept_ridge}")
print(f"Mean RMSE (Ridge Regression): {mean_rmse_ridge}")
