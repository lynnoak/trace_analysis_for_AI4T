import re
import pandas as pd

# data_raw = pd.read_csv('./Luxembourg/data_raw.csv')
# read raw data
try:
    data_raw = pd.read_csv('data_raw.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()


data = data_raw.copy()
data['User_id'] = data['User ID']
# regular expressions
patterns = {
    'module_id': r"course module id '(\d+)'",
    'chapter_id': r"chapter with id '(\d+)'",
    'h5p_id': r"H5P with the id '(\d+)'",
    'scorm_id': r"SCORM with the id (\d+)"
}

# separate the description
def parse_description(description):
    description = str(description)
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, description)
        if match:
            result[key] = match.group(1)
        else:
            result[key] = None
    return result


columns = ['Course_module_id', 'Chapter_id', 'H5P_id', 'SCORM_id']
data[columns] = data['Description'].apply(parse_description).apply(pd.Series)
data['Resource_id'] = data[['Course_module_id', 'Chapter_id', 'H5P_id', 'SCORM_id']].astype(str).apply(lambda x: '_'.join(x.dropna()), axis=1)



# calculate the duration
data['Time'] = pd.to_datetime(data['Time'], format='%d/%m/%y, %H:%M')

def CalculateDuration(group):
    group = group.sort_values('Time')
    duration = (group['Time'].shift(-1) - group['Time']).dt.total_seconds()
    duration[duration <= 0] = 10
    duration[duration >600] = 10

    return duration

data['Duration'] = data.groupby('User_id').apply(CalculateDuration).reset_index(drop=True)
data['Duration'] = data['Duration'].fillna(0)

# define the Events
data['Event'] = 'Other'
data.loc[data['Event name'] == 'Course viewed', 'Event'] = 'Course viewed'
data.loc[data['Event name'] == 'Chapter viewed', 'Event'] = 'Chapter viewed'
data.loc[data['Event name'] == 'Course module viewed', 'Event'] = data.loc[data['Event name'] == 'Course module viewed', 'Component'] + ' viewed'
data.loc[data['Event name'] == 'H5P content viewed', 'Event'] = 'H5P content viewed'

#define the Events type
data['Event_type'] = 'Other'
data.loc[data['Event'] == 'Chapter viewed', 'Event_type'] = 'Reading'
data.loc[data['Event'] == 'H5P content viewed', 'Event_type'] = 'Activity'
data.loc[data['Event'] == 'H5P action', 'Event_type'] = 'Activity'

# Module define for Luxembourg

data['Module'] = data['Event context']
data.loc[~data['Module'].str.contains('Module'), 'Module'] = 'Other'

data.to_csv('data.csv',index=False)

performance = pd.DataFrame()
performance['count_h5p'] = data[data['H5P_id'].notnull()].groupby('User_id')['H5P_id'].count()
performance['duration_h5p'] = data[data['H5P_id'].notnull()].groupby('User_id')['Duration'].mean()

performance.to_csv('performance.csv')