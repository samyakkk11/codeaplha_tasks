import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download the dataset
path = kagglehub.dataset_download("gokulrajkmv/unemployment-in-india")
print("Dataset downloaded to:", path)

# Load the dataset
file_path = f"{path}/Unemployment in India.csv"
df = pd.read_csv(file_path)

# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Basic information about the dataset
print("\nDataset information:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data Cleaning (if necessary)
# For example, dropping rows with missing values
df = df.dropna()

# Visualizing the unemployment rate over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Boxplot to see the distribution of unemployment rates
plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='Estimated Unemployment Rate (%)', data=df)
plt.title('Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap (only for numeric columns)
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Columns Only)')
plt.show()