import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import os
os.makedirs("figures", exist_ok=True)

california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value (in $100,000)")
plt.ylabel("Frequency")
plt.savefig("figures/target_distribution.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.savefig("figures/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(X["MedInc"], y, alpha=0.5)
plt.title("Median Income vs Median House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value (in $100,000)")
plt.savefig("figures/medinc_scatter.png")
plt.close()
