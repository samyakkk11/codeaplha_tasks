import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = "data/carprice.csv"
df = pd.read_csv(file_path)

# Data Cleaning (if necessary)
df = df.dropna()

# --- Enhanced Exploratory Data Analysis (EDA) ---

# 1. Distribution of Selling Price (Target Variable)
plt.figure(figsize=(12, 7))
sns.histplot(df['Selling_Price'], bins=30, kde=True, color='#66b3ff', edgecolor='black')
plt.title('Selling Price Distribution: Most Cars in Lower Price Ranges', fontsize=18, fontweight='bold')
plt.xlabel('Selling Price (Lakhs)', fontsize=14)
plt.ylabel('Number of Cars', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.text(df['Selling_Price'].max() * 0.7, plt.gca().get_ylim()[1] * 0.8,
         "Most cars priced\nunder 10 lakhs", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# 2. Selling Price vs. Present Price
plt.figure(figsize=(12, 7))
sns.scatterplot(x='Present_Price', y='Selling_Price', data=df, color='#4CAF50', s=80, alpha=0.7)
plt.title('Selling Price vs. Present Price: Clear Positive Correlation', fontsize=18, fontweight='bold')
plt.xlabel('Present Price (Lakhs)', fontsize=14)
plt.ylabel('Selling Price (Lakhs)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.annotate("Strong Positive\nRelationship", xy=(df['Present_Price'].median(), df['Selling_Price'].median()),
             xytext=(df['Present_Price'].max() * 0.6, df['Selling_Price'].max() * 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)
plt.show()

# 3. Selling Price vs. Year
plt.figure(figsize=(12, 7))
sns.boxplot(x='Year', y='Selling_Price', data=df, palette='viridis')
plt.title('Selling Price by Year: Newer Cars Fetch Higher Prices', fontsize=18, fontweight='bold')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Selling Price (Lakhs)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 4. Selling Price vs. Fuel Type
plt.figure(figsize=(12, 7))
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=df, palette='coolwarm')
plt.title('Selling Price by Fuel Type: Diesel Cars Command Higher Prices', fontsize=18, fontweight='bold')
plt.xlabel('Fuel Type', fontsize=14)
plt.ylabel('Selling Price (Lakhs)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Feature Engineering
df = pd.get_dummies(df, drop_first=True)
target_column = 'Selling_Price'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred_linear = linear_model.predict(X_test)
print("\nLinear Regression Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_linear))
print("R^2 Score:", r2_score(y_test, y_pred_linear))

y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Regressor Results:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_rf))
print("R^2 Score:", r2_score(y_test, y_pred_rf))

# Feature Importance (Random Forest)
feature_importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances (Random Forest):")
print(importance_df)

# Plot feature importances (Enhanced)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', hue='Feature', legend=False)
plt.title('Feature Importance in Predicting Selling Price', fontsize=18, fontweight='bold')
plt.xlabel('Importance Score', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()