import re
import pandas as pd

# read raw data
try:
    data_raw = pd.read_csv('data_raw.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

data = data_raw.copy()
data['User_id'] = data['User full name']
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

data['Time'] = pd.to_datetime(data['Time'], format='%d/%m/%y, %H:%M:%S')
def CalculateDuration(group):
    group = group.sort_values('Time')
    duration = (group['Time'].shift(-1) - group['Time']).dt.total_seconds()
    duration[duration <= 0] = 1
    duration[duration >600] = 10

    return duration

data['Duration'] = data.groupby('User_id').apply(CalculateDuration).reset_index(drop=True)
data['Duration'] = data['Duration'].fillna(0)

# define the Events
data['Event'] = 'Other'
data.loc[data['Event name'] == 'Course viewed', 'Event'] = 'Course viewed'
data.loc[data['Event name'] == 'Chapter viewed', 'Event'] = 'Chapter viewed'
data.loc[data['Event name'] == 'Course module viewed', 'Event'] = data.loc[data['Event name'] == 'Course module viewed', 'Component'] + ' viewed'
data.loc[data['Event name'] == 'SCORM viewed', 'Event'] = 'SCORM viewed'
data.loc[data['Event name'] == 'H5P content viewed', 'Event'] = 'H5P viewed'

#define the Events type
data['Event_type'] = 'Other'
data.loc[data['Event'] == 'Chapter viewed', 'Event_type'] = 'Reading'
data.loc[data['Event'] == 'SCORM viewed', 'Event_type'] = 'Activity'
data.loc[data['Event'] == 'SCORM package viewed', 'Event_type'] = 'Activity'
data.loc[data['Event'] == 'H5P content viewed', 'Event_type'] = 'Activity'
data.loc[data['Event'] == 'H5P viewed', 'Event_type'] = 'Activity'

# Module define for Ireland

data['Module'] = 'Other'
data.loc[data['Event context'] == 'Book: General Presentation - book', 'Module'] = 'Module 0'
data.loc[data['Event context'] == 'Book: Module 1 - first book', 'Module'] = 'Module 1'
data.loc[data['Event context'] == 'Book: Module 1 - second book', 'Module'] = 'Module 1'
data.loc[data['Event context'] == 'Book: Module 2 - first book', 'Module'] = 'Module 2'
data.loc[data['Event context'] == 'Book: Module 2 - second book', 'Module'] = 'Module 2'
data.loc[data['Event context'] == 'Book: Module 3 - book', 'Module'] = 'Module 3'
data.loc[data['Event context'] == 'Book: Module 4 - book', 'Module'] = 'Module 4'
data.loc[data['Event context'] == 'Book: Conclusion - book', 'Module'] = 'Module 7'
data.loc[data['Event context'] == 'SCORM package: Activity 1.2.1', 'Module'] = 'Module 1'
data.loc[data['Event context'] == 'SCORM package: 2.2.4 Activity: The origin of 3 AI technologies', 'Module'] = 'Module 2'

data.to_csv('data.csv',index=False)
