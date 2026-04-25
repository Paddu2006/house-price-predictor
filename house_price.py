# House Price Predictor
# By Padma Shree
# Project 20 of 25

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

# Step 1 - Load data
print("=== HOUSE PRICE PREDICTOR ===")
df = pd.read_csv(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\05_resources\datasets\house_prices.csv")

print("Total houses:", len(df))
print("Columns:", df.columns.tolist())
print("\nPrice range: $", df["price"].min(), "to $", df["price"].max())
print("Average price: $", round(df["price"].mean(), 2))

# Step 2 - Prepare data
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
            "floors", "waterfront", "view", "condition", "grade",
            "sqft_above", "yr_built", "zipcode"]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 3 - Train Linear Regression
print("\n=== LINEAR REGRESSION ===")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"MAE: ${lr_mae:,.2f}")
print(f"R2 Score: {lr_r2:.4f}")

# Step 4 - Train Random Forest
print("\n=== RANDOM FOREST ===")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"MAE: ${rf_mae:,.2f}")
print(f"R2 Score: {rf_r2:.4f}")

print(f"\nBest model: {'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'}")

# Step 5 - Charts
# Chart 1 - Price distribution
plt.figure(figsize=(12,6))
plt.hist(df["price"], bins=50, color="blue", alpha=0.7)
plt.title("House Price Distribution")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\house_price_predictor\price_distribution.png")
plt.show()
print("Chart 1 saved!!")

# Chart 2 - Actual vs Predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, rf_pred, alpha=0.5, color="green")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\house_price_predictor\actual_vs_predicted.png")
plt.show()
print("Chart 2 saved!!")

# Chart 3 - Feature importance
feature_importance = pd.Series(rf.feature_importances_, index=features)
feature_importance.sort_values().plot(kind="barh", color="orange", figsize=(10,6))
plt.title("Features That Most Influence House Price")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(r"C:\Users\Padma shree jena\Desktop\PadduDS_Journey\04_capstone\house_price_predictor\feature_importance.png")
plt.show()
print("Chart 3 saved!!")

print("\nHouse price prediction complete!!")