import pandas as pd

# read raw data
# data_raw = pd.read_csv('./Italy/data_raw.csv')
try:
    data_raw = pd.read_csv('data_raw.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

data = data_raw.copy()
data['User_id'] = data['user_id']
data['Resource_id'] = data['resource_id']
data = data.dropna(subset=['User_id','Resource_id'])

# calculate the duration
data['Time'] = pd.to_datetime(data['timestamp'], format='%Y/%m/%d %H:%M')

def CalculateDuration(group):
    group = group.sort_values('Time')
    duration = (group['Time'].shift(-1) - group['Time']).dt.total_seconds()
    duration[duration <= 0] = 10
    duration[duration >600] = 10

    return duration

data['Duration'] = data.groupby('User_id').apply(CalculateDuration).reset_index(drop=True)
data['Duration'] = data['Duration'].fillna(0)

#link to the mooc
# correspondence = pd.read_csv('./Italy/correspondence.csv')
try:
    correspondence = pd.read_csv('correspondence.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

data = pd.merge(data,correspondence,how='left',left_on = 'Resource_id',right_on= 'Resource ID')

# define the Events
data['Event'] = data['UID']

#define the Events type
data['Event_type'] = data['UID'].astype(str).str[-1]
data['Event_type'] = data['Event_type'].map({'a':'Activity','t':'Reading','v':'Other','n':'Other'})

data['Chapter_id'] = data.apply(lambda row: row['Resource_id'] if row['Event_type'] == 't' else None, axis=1)

#define the Module
data['Module'] = "Module " + data['Resource_id'].astype(str).str[0]

data.to_csv('data.csv',index=False)

performance = pd.DataFrame()
performance['Answered'] = data[data['verb_name'] == 'answered'].groupby('User_id').size()
performance['Completed'] = data[(data['verb_name'] == 'completed') & (data['Event_type'] == 'Activity')].groupby('User_id').size()

performance.to_csv('performance.csv')