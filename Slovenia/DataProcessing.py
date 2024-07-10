import re
import pandas as pd

# read raw data
# data_raw = pd.read_csv('./Slovenia/data_raw.csv')
try:
    data_raw = pd.read_csv('data_raw.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

data = data_raw.copy()
# regular expressions
patterns = {
    'user_id': r"user with id '(\d+)'",
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


columns = ['User_id', 'Course_module_id', 'Chapter_id', 'H5P_id', 'SCORM_id']
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
data.loc[data['Event name'] == 'xAPI statement received', 'Event'] = 'H5P action'
data.loc[data['Event name'] == 'Course module viewed', 'Event'] = data.loc[data['Event name'] == 'Course module viewed', 'Component'] + ' viewed'
data.loc[data['Event name'] == 'H5P content viewed', 'Event'] = 'H5P viewed'

#define the Events type
data['Event_type'] = 'Other'
data.loc[data['Event'] == 'Chapter viewed', 'Event_type'] = 'Reading'
data.loc[data['Event'] == 'H5P viewed', 'Event_type'] = 'Activity'
data.loc[data['Event'] == 'H5P action', 'Event_type'] = 'Activity'

try:
    correspond = pd.read_csv('correspond.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

t1 = correspond[['Module','ID']].drop_duplicates()
t2 = correspond[['Module','h5p moodle module id']].drop_duplicates()
t2.columns = t1.columns
id2module = pd.concat([t1,t2]).dropna()
id2module['ID'] = id2module['ID'].astype(int).astype(str)
data = pd.merge(left=data, right=id2module, how='left', left_on='Course_module_id', right_on='ID')
data = data.drop('ID',axis=1)

t3 = correspond[['Module','h5p id']].drop_duplicates()
t3.dropna(subset=['h5p id'], inplace=True)
t3['h5p id'] = (t3['h5p id'] + 700).astype(int).astype(str)

filtered_rows = data[data['H5P_id'].notna()].reset_index(drop=True)
merged_data = pd.merge(filtered_rows, t3, left_on='H5P_id', right_on='h5p id', how='left')
empty_module_index = filtered_rows[filtered_rows['Module'].isna()].index
data.loc[empty_module_index, 'Module'] = merged_data.loc[empty_module_index, 'Module_y']
data['Module'] = data['Module'].fillna('Other')

data.to_csv('data.csv',index=False)
