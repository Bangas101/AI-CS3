# Group 2: Wholesale Customers Clustering (K-means vs DBSCAN)

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import kruskal, chi2_contingency
sns.set(style='whitegrid')

# Part A: Data Loading & Preprocessing

file_path = r"./Wholesale customers data.csv"

df = pd.read_csv(file_path)
print('Initial shape:', df.shape)
print(df.dtypes)

# Handle missing & duplicates
# Using median imputation for any potential missing values and dropping duplicates
df = df.fillna(df.median(numeric_only=True)).drop_duplicates()
print('Shape after cleaning:', df.shape)

# Scale numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[num_cols])

# EDA
print(df.describe())

# Part B: K-Means Clustering

# Elbow Method to find optimal K (range 2 to 10)
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Calculate silhouette score for K=2 to K=10
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for K-Means')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

# Based on analysis, K=2 is often selected.
k_optimal = 2
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto').fit(X_scaled)
k_labels = kmeans.labels_
df['KMeans_Cluster'] = k_labels
print(f"Optimal K-Means Silhouette Score (K={k_optimal}): {silhouette_score(X_scaled, k_labels):.4f}")
print(f"Optimal K-Means Davies-Bouldin Score (K={k_optimal}): {davies_bouldin_score(X_scaled, k_labels):.4f}")

# Part C: DBSCAN Clustering

# Find optimal epsilon (eps) using Nearest Neighbors
neigh = NearestNeighbors(n_neighbors=2).fit(X_scaled)
distances, indices = neigh.kneighbors(X_scaled)
distances = np.sort(distances[:, 1], axis=0)

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title('K-distance Graph for DBSCAN (k=2)')
plt.xlabel('Data points sorted by distance')
plt.ylabel('Epsilon (eps)')
plt.show()

# Optimal parameters based on common practice or previous analysis: eps=0.5, min_samples=5
eps_optimal = 0.5
min_samples_optimal = 5
dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_optimal).fit(X_scaled)
db_labels = dbscan.labels_
df['DBSCAN_Cluster'] = db_labels
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)

print(f"DBSCAN found {n_clusters_db} clusters and {n_noise} noise points.")

# DBSCAN Evaluation (excluding noise points -1)
if n_clusters_db > 1:
    db_silhouette_score = silhouette_score(X_scaled[db_labels != -1], db_labels[db_labels != -1])
    db_davies_bouldin_score = davies_bouldin_score(X_scaled[db_labels != -1], db_labels[db_labels != -1])
else:
    db_silhouette_score = 'N/A'
    db_davies_bouldin_score = 'N/A'

print(f"DBSCAN Silhouette Score (excluding noise): {db_silhouette_score}")
print(f"DBSCAN Davies-Bouldin Score (excluding noise): {db_davies_bouldin_score}")


# Part D: Cluster Analysis

# Function to perform Kruskal-Wallis test (for numeric features)
def kruskal_wallis_test(df, column, cluster_col):
    groups = [group[column].values for name, group in df.groupby(cluster_col)]
    if len(groups) > 1:
        stat, p = kruskal(*groups)
        return p
    return 1 # Not significant if only one group

# Function to perform Chi-squared test (for categorical features)
def chi_squared_test(df, column, cluster_col):
    contingency_table = pd.crosstab(df[column], df[cluster_col])
    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
        stat, p, dof, expected = chi2_contingency(contingency_table)
        return p
    return 1

# Analyze K-Means Clusters
print("\n--- K-Means Cluster Analysis (K=2) ---")
k_cluster_summary = df.groupby('KMeans_Cluster')[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].mean().round(0)
print("Mean Spend per Cluster:")
print(k_cluster_summary)

print("\nStatistical Significance (Kruskal-Wallis P-values for numeric features):")
k_p_values = {}
for col in ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']:
    k_p_values[col] = kruskal_wallis_test(df, col, 'KMeans_Cluster')
print(pd.Series(k_p_values))

print("\nStatistical Significance (Chi-squared P-values for Channel/Region):")
k_p_values_cat = {}
for col in ['Channel', 'Region']:
    k_p_values_cat[col] = chi_squared_test(df, col, 'KMeans_Cluster')
print(pd.Series(k_p_values_cat))

# Analyze DBSCAN Clusters (excluding noise)
print("\n--- DBSCAN Cluster Analysis ---")
db_cluster_summary = df[df['DBSCAN_Cluster'] != -1].groupby('DBSCAN_Cluster')[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']].mean().round(0)
print("Mean Spend per Cluster (Noise excluded):")
print(db_cluster_summary)

# Part E: Compare and Select

print("\n--- Clustering Comparison (K-Means vs DBSCAN) ---")
comparison = pd.DataFrame({
    'Metric': ['Silhouette Score', 'Davies-Bouldin Score', 'Number of Clusters (excluding noise)'],
    'K-Means': [silhouette_score(X_scaled, k_labels), davies_bouldin_score(X_scaled, k_labels), k_optimal],
    'DBSCAN': [db_silhouette_score, db_davies_bouldin_score, n_clusters_db]
})

print(comparison)


# Part F: Visualization & Reflection

# Apply PCA for 2D visualization
pca=PCA(2).fit_transform(X_scaled)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); sns.scatterplot(x=pca[:,0],y=pca[:,1],hue=k_labels,palette='tab10'); plt.title('K-means PCA')
plt.subplot(1,2,2); sns.scatterplot(x=pca[:,0],y=pca[:,1],hue=db_labels,palette='tab10'); plt.title('DBSCAN PCA'); plt.show()

print('''
Reflection:
K-means assumes spherical clusters and equal variance.
DBSCAN finds arbitrary shapes but depends on eps/min_samples.
Use clusters to identify high TotalSpend customers for targeted marketing.''')