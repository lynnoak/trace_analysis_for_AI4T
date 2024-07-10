import numpy as np
import pandas as pd
import json
import subprocess

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from collections import Counter

import matplotlib.pyplot as plt
colors = plt.cm.get_cmap('tab10', 10)(range(10))

#localpath = './Ireland/'
#localpath = './Luxembourg/'
localpath = './Slovenia/'
#localpath = './Italy/'
#localpath = './France/'


if __name__ == "__main__":
    try:
        subprocess.run(['python', 'ProfileProcessing.py', localpath])
        print('Profile processed successfully')
    except:
        print('Error with Profile processing')

# read data

try:
    profile = pd.read_csv(localpath+'profile.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()
profile.index = profile['User_id']
profile.index.name = 'User_id'
profile.drop(['User_id'],axis = 1,inplace =True)

with open(localpath+'indicators.json', 'r') as f:
    indicators = json.load(f)

indicators_list = list(indicators.keys())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(profile)
profile_scaled = pd.DataFrame(scaled_data, index=profile.index, columns=profile.columns)

def myClustering(X,title = localpath):

    # Outlier detection and PCA for show
    model = IsolationForest(contamination=0.05, random_state=42)
    pca = PCA(n_components=2)

    X = np.array(X)
    X_pca = pca.fit_transform(X)

    # separate outliers and non-outliers
    outliers = model.fit_predict(X)
    outliers_index = np.where(outliers == -1)
    inliers_index = np.where(outliers == 1)
    inliers = X[inliers_index]

    # kmeans with silhouette_score metric
    best_k = None
    best_silhouette = -1
    consecutive_no_improvement = 0

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(inliers)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(inliers, labels)

        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_k = k
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= 5:
            break

    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(inliers)

    k_label = kmeans.labels_
    counter = Counter(k_label)
    rank = {value: rank for rank, (value, count) in enumerate(counter.most_common(), 0)}
    k_label = np.array([rank[value] for value in k_label])

    labels = np.full(X.shape[0], -1)
    labels[inliers_index] = k_label

    # assign colors to inliers and outliers
    color_map = np.zeros((len(labels), 4))
    color_map[inliers_index] = colors[labels[inliers_index]]
    color_map[outliers_index] = [0, 0, 0, 1]  # Black color for outliers

    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_map)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.text(0.05, 0.95, f'Clusters: {best_k}', transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.savefig(title + 'Cluster.png')
    #plt.show()

    return labels,best_k,X_pca

labels,best_k,X_pca = myClustering(profile_scaled)
profile["Label"] = labels

def Calculate_Indicators(user):
    result = []
    for i in indicators.keys():
        result.append(np.mean(user[indicators[i]]))
    return result

profile_indicators = pd.DataFrame()
profile_indicators[indicators_list] = profile_scaled.apply(Calculate_Indicators,axis=1).apply(pd.Series)
profile_indicators["Label"] = profile["Label"]
filtered_profile_indicators = profile_indicators[profile_indicators["Label"] != -1]
min_values = filtered_profile_indicators.iloc[:,:-1].min()
max_values = filtered_profile_indicators.iloc[:,:-1].max()
profile_indicators.iloc[:,:-1] = (profile_indicators.iloc[:,:-1] - min_values) / (max_values - min_values)* 0.9 + 0.1


profile_indicators.to_csv(localpath+'profile_indicators.csv')

def show_indicators(x,title=localpath+'test.png',color = 'blue'):
    features = ['Engagement', 'Completion', 'Curiosity', 'Performance', 'Reactivity', 'Regularity']
    values = x[:6].values.tolist()
    values = [round(value, 3) for value in values]

    angles = [n / float(len(features)) * 2 * 3.1415926 for n in range(len(features))]
    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

    ax.plot(angles, values, linewidth=1, linestyle='solid', color=color)
    ax.fill(angles, values, color=color, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim([0, 1])


    for i, angle in enumerate(angles[:-1]):
        x = angle
        y = values[i]
        ax.text(x, y, values[i], ha='center', va='center')

    plt.savefig(title)
    #plt.show()

for i in range(best_k):
    cluster_i = profile_indicators[profile_indicators['Label'] == i]
    centroid_i = cluster_i.mean()
    show_indicators(centroid_i,(localpath+'Description Card for Clustering '+str(i)+' .png'),color = colors[i])


correlation = profile.corrwith(profile_indicators["Label"]).abs()
strong_correlation = correlation[correlation > correlation.mean()]
print(strong_correlation)
strong_correlation.to_csv(localpath+'strong_corr_features.csv')
correlation = profile_indicators.corrwith(profile_indicators["Label"]).abs()
strong_correlation = correlation[correlation > correlation.mean()]
print(strong_correlation)
strong_correlation.to_csv(localpath+'strong_corr_indicators.csv')