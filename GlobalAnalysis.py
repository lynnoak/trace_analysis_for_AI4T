import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from ProfileAnalysis import myClustering,show_indicators,colors

localpath = 'Global Analysis 3.0'
list_countries = ['Ireland', 'Luxembourg', 'Slovenia','Italy','France']

data_module = {}
Time_spend = pd.DataFrame()
Number_of_Actions = pd.DataFrame()

for country in list_countries:
    data_module[country] = pd.read_csv('./' + country + '/Module.csv')
    data_module[country] = data_module[country].set_index(data_module[country]['Module'])
    data_module[country].drop('Module', axis=1, inplace=True)
    Time_spend[country] = data_module[country]['Time spent']
    Number_of_Actions[country] = data_module[country]['Number of Actions']


data = {}

for country in list_countries:
    data[country] = pd.read_csv('./' + country + '/profile_indicators.csv')
    data[country] = data[country].set_index(data[country]['User_id'])
    data[country].drop('User_id', axis=1, inplace=True)
    data[country]['Country'] = country

data_global = pd.concat(data.values())

# one hot for represent the country
one_hot = pd.get_dummies(data_global['Country'])
profile = pd.concat([data_global.iloc[:,:6], one_hot], axis=1)

#standard scaler again for global
scaler = StandardScaler()
scaled_data = scaler.fit_transform(profile.iloc[:,:6])
profile_scaled = profile.copy()
profile_scaled.iloc[:,:6] = scaled_data
country_color = {country: i for i, country in enumerate(list_countries)}
profile_scaled['color'] = data_global['Country'].map(country_color)

#clustering
labels,best_k,X_pca = myClustering(scaled_data,localpath)
data_global["Global_Label"] = labels


# show the data points for each country
color_map_country = colors[profile_scaled['color']]
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color_map_country)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
legend_labels = []
for country, color in zip(list_countries, colors):
    legend_labels.append(ax.scatter([], [], color=color, label=country))
ax.legend(handles=legend_labels, loc='upper right')
plt.savefig(localpath + 'Country.png')
#plt.show()

profile_indicators = data_global.copy()
filtered_profile_indicators = profile_scaled[profile_indicators["Global_Label"] != -1]
min_values = filtered_profile_indicators.iloc[:,:6].min()
max_values = filtered_profile_indicators.iloc[:,:6].max()
profile_indicators.iloc[:,:6] = (profile_scaled.iloc[:,:6] - min_values) / (max_values - min_values)* 0.9 + 0.1
profile_indicators.to_csv(localpath+'profile_indicators.csv')

#Show the description cards

for i in range(best_k):
    cluster_i = profile_indicators[profile_indicators['Global_Label'] == i]
    centroid_i = cluster_i.iloc[:,:6].mean()
    show_indicators(centroid_i,(localpath+'Description Card for Clustering '+str(i)+' .png'),colors[i])


correlation = data_global.iloc[:,:6].corrwith(data_global["Global_Label"]).abs()
strong_correlation = correlation[correlation > correlation.mean()]
print(strong_correlation)
strong_correlation.to_csv(localpath+'strong_corr_indicators.csv')

"""
def show_several(data, titles):
    features = ['Engagement', 'Completion', 'Curiosity', 'Performance', 'Reactivity', 'Regularity']
    angles = [n / float(len(features)) * 2 * 3.1415926 for n in range(len(features))]
    angles += angles[:1]
    data = np.array(data)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})

    for i, (x, title) in enumerate(zip(data, titles)):
        values = x[:6].tolist()
        values = [round(value, 3) for value in values]
        values += values[:1]

        color = plt.cm.get_cmap('tab10', 10)(i)

        ax.plot(angles, values, linewidth=1, linestyle='solid', color=color, alpha=0.3)
        ax.fill(angles, values, color, alpha=0.3)

        for j, angle in enumerate(angles[:-1]):
            x = angle
            y = values[j]
            ax.text(x, y, values[j], ha='center', va='center')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim([0, 1])

    plt.savefig(localpath + 'test.png')
    plt.show()

show_several(data_global.iloc[1:3,:],titles=[1,2])
"""
