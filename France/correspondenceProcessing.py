import pandas as pd
import chardet

file_path = './France/correspondence 1.csv'
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']

data1 = pd.read_csv(file_path, encoding=encoding)

data1.columns = ['meta', 'Module', 'Chapter_id', 'Resource_id', 'c1', 'c2', 'c3', 'c4', 'm1', 'm2']
data1c = data1[['meta', 'Module', 'Chapter_id', 'Resource_id', 'c1', 'c2', 'c3', 'c4']]
datam = data1[['meta', 'Module', 'Chapter_id', 'Resource_id', 'm1', 'm2']]

datam = pd.melt(datam, id_vars=['meta', 'Module', 'Chapter_id', 'Resource_id'], value_vars=['m1', 'm2'], var_name='m', value_name='Marsha_id')
datam = datam.dropna(subset=['Marsha_id'])
datam['Module'] = datam['meta'].str[12].apply(lambda x: 'Module ' + x)
datam['Event'] = 'Video'
datam = datam.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

datam.to_csv('./France/correspondence video.csv', index=False)


data1 = pd.melt(data1c, id_vars=['meta', 'Module', 'Chapter_id', 'Resource_id'], value_vars=['c1', 'c2', 'c3', 'c4'], var_name='c', value_name='Com_id')
data1 = data1.dropna(subset=['Com_id'])
data1['Module'] = data1['meta'].str[12].apply(lambda x: 'Module ' + x)
data1['Event'] = data1['meta'].str[-1].map({'v': 'Video', 't': 'Reading', 'a': 'Activity'})
data1 = data1.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


data2 = pd.read_csv('./France/correspondence 2.csv', encoding=encoding)
data2.columns = ['Module', 'Module_id', 'Chapter_id', 'Resource_id', 'c1', 'c2']
data2 = pd.melt(data2, id_vars=['Module', 'Module_id', 'Chapter_id', 'Resource_id'], value_vars=['c1', 'c2'], var_name='c', value_name='Com_id')
data2 = data2.dropna(subset=['Com_id'])
data2['Event'] = 'Other'
data2 = data2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


common_columns = ['Com_id', 'Resource_id', 'Chapter_id', 'Module', 'Event']
data = pd.concat([data1[common_columns], data2[common_columns]], ignore_index=True)
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

data.to_csv('./France/correspondence.csv', index=False)
