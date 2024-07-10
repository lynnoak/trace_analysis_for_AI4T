import chardet
import pandas as pd

file_path = './France/correspondence 1.csv'
with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']

common_columns = ['User_id','Resource_id','Time','Duration','Event','Event_type','Chapter_id','Module']

data_video = pd.read_csv('./France/video_played.csv')
data_video.columns = ['Time','Duration','User_id','Marsha_id']
cor_video = pd.read_csv('./France/correspondence video.csv')
data_1 = data_video.merge(cor_video,on='Marsha_id',how='left')
data_1['Event_type'] = 'Other'
data_1['Time'] = pd.to_datetime(data_1['Time'])
data_1['Duration']=data_1['Duration'].astype('float')
data_1 = data_1[common_columns]

data_2 = pd.read_csv('./France/page_close.csv')
data_2.columns = ['User_id','Time','Page']
cor_ = pd.read_csv('./France/correspondence.csv')
Chapter_id_list = cor_['Chapter_id'].unique()

def extract_chapter_id(page):
    for chapter_id in Chapter_id_list:
        if chapter_id in page:
            return chapter_id
    return None

data_2['Chapter_id'] = data_2['Page'].apply(extract_chapter_id)
cor_page = cor_.drop_duplicates(subset=['Chapter_id'],keep='first')
data_2 = data_2.merge(cor_page,on='Chapter_id',how='left')
data_2['Event_type'] = 'Reading'
data_2['Time'] = pd.to_datetime(data_2['Time'])
data_2['Duration'] = 0
data_2 = data_2[common_columns]

data_3 = pd.read_csv('./France/problem_check.csv')
data_3 = data_3.rename(columns={'username': 'User_id', 'time': 'Time', 'event.problem_id': 'Com_id'})

performance = pd.DataFrame()
performance['Number'] = data_3.groupby('User_id').size()
performance['Quizze_number'] = data_3.groupby('User_id')['Com_id'].nunique()
performance['Aveage_grade'] = data_3.groupby('User_id')['event.grade'].mean()
performance['Correct_rate'] = data_3[data_3['event.success'] == 'correct'].groupby('User_id').size()
performance['Correct_rate'] = performance['Correct_rate'] / performance['Number']
performance.to_csv('./France/performance.csv')

data_3 = data_3.merge(cor_,on='Com_id',how='left')
data_3['Event_type'] = 'Activity'
data_3['Time'] = pd.to_datetime(data_3['Time'])
data_3['Duration'] = 0
data_3 = data_3[common_columns]

data = pd.concat([data_1,data_2,data_3])

def CalculateDuration(group):
    group = group.sort_values('Time')
    duration = (group['Time'].shift(-1) - group['Time']).dt.total_seconds()
    duration[duration <= 0] = 10
    duration[duration >600] = 10

    return duration

data['Duration_'] = data.groupby('User_id').apply(CalculateDuration).reset_index(drop=True)
data['Duration_'] = data['Duration_'].fillna(0)
filter = data['Duration'] == 0
data.loc[filter, 'Duration'] = data.loc[filter, 'Duration_']
data = data.dropna()
data = data[common_columns]
data.to_csv('./France/data.csv', index=False)