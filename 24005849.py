import pandas as pd
import os 
os.environ ['OMP_NUM_THREADS'] = '1'


# Load your dataset
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Mall_Customers.csv")

# Preview the data
print(df.head())
print(df.describe())

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow method to find optimal clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid()
plt.show()

from sklearn.linear_model import LinearRegression

# Prepare data
X = df[['Age']].values  # Independent variable
y = df['Spending Score (1-100)'].values  # Dependent variable

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Scatter plot with regression line
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Regression Line Fit')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()


# Histogram for 'Age'
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

import seaborn as sns
import numpy as np

# Compute correlation matrix
corr_matrix = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()

# Heatmap to visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()