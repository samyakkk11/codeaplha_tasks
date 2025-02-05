import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Advertising.csv', header=None)  # Assuming no header row

# Rename columns based on your data
data.columns = ['Index', 'TV', 'Radio', 'Newspaper', 'Sales']

# 1. Distribution of Sales Prices
plt.figure(figsize=(10, 6))
sns.histplot(data['Sales'], bins=30, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# 2. Scatter Plot of Sales Price vs. TV
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TV', y='Sales', data=data)
plt.title('Sales vs. TV')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()

# 3. Scatter Plot of Sales Price vs. Radio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Radio', y='Sales', data=data)
plt.title('Sales vs. Radio')
plt.xlabel('Radio')
plt.ylabel('Sales')
plt.show()

# 4. Scatter Plot of Sales Price vs. Newspaper
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Newspaper', y='Sales', data=data)
plt.title('Sales vs. Newspaper')
plt.xlabel('Newspaper')
plt.ylabel('Sales')
plt.show()

# 5. Correlation Matrix (heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()