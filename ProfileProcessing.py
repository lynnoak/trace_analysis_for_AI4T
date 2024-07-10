import numpy as np
import pandas as pd
import sys

#localpath = './Ireland/'
#localpath = './Luxembourg/'
localpath = './Slovenia/'
#localpath = './Italy/'
#localpath = './France/'

try:
    localpath = sys.argv[1]
    print('Local path from sys')
except:
    print('Did not read local path from sys')

# read data
# require data features: 'User_id','Resource_id','Time','Duration','Event','Event_type','Chapter_id','Module'
try:
    data = pd.read_csv(localpath+'data.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

data['Time'] = pd.to_datetime(data['Time'])

# Event_list is a global variable
Event_list = data['Event_type'].unique()

#Calculation of Engagement

def Calculate_Time(user):
    Days = user['Time'].dt.date
    Days = len(Days.unique())
    Duration_total = user['Duration'].sum()
    Counts_total = len(user)
    return [Days, Duration_total, Counts_total]

profile = data.groupby('User_id').apply(Calculate_Time).apply(pd.Series)
profile.index.name = 'User_id'
profile.columns = ['Days', 'Duration_total', 'Counts_total']

Event_list_counts = ['Counts '+i for i in Event_list]
profile[Event_list_counts] = data.pivot_table(index='User_id', columns='Event_type', values='Time', aggfunc='count', fill_value=0)

Event_list_duration = ['Duration '+i for i in Event_list]
profile[Event_list_duration] = data.pivot_table(index='User_id', columns='Event_type', values='Duration', aggfunc='sum', fill_value=0)

Engagement_list = ['Counts_total','Duration_total','Counts Activity','Duration Activity','Counts Reading','Duration Reading']

#Calculation of Completion

Number_Reading = data[data['Event_type'] == 'Reading']['Chapter_id'].nunique()
Number_Activity = data[data['Event_type'] == 'Activity']['Resource_id'].nunique()
Number_Other = data[data['Event_type'] == 'Other']['Resource_id'].nunique()
Number_Module = data['Module'].nunique()

profile['Rate_Reading'] = data.groupby('User_id')['Chapter_id'].nunique()/Number_Reading
profile['Rate_Activity'] = data[data['Event_type'] == 'Activity'].groupby('User_id')['Resource_id'].nunique()/Number_Activity
profile['Rate_Other'] = data[data['Event_type'] == 'Other'].groupby('User_id')['Resource_id'].nunique()/Number_Other
profile['Rate_Module'] = data.groupby('User_id')['Module'].nunique()/Number_Module
profile = profile.fillna(0)

Completion_list = ['Rate_Reading','Rate_Activity','Rate_Other','Rate_Module']

#Calculation of Curiosity

profile['Nunique_Action'] = data.groupby('User_id')['Event'].nunique()
Module_Duration = data.pivot_table(index='User_id', columns='Module', values='Duration', aggfunc='sum', fill_value=0)

common_behavior = data.groupby(['Event', 'Resource_id']).filter(lambda x: len(x) >= 0.8 * data['User_id'].nunique())
user_common_behavior_count = common_behavior.groupby('User_id').size()
def Calculate_Different(user):
    if user.name in list(user_common_behavior_count.index):
        return len(user)-user_common_behavior_count[user.name]
    else:
        return 0

profile['Different_Count'] = data.groupby('User_id').apply(Calculate_Different)

profile['Module_Average_Duration'] = Module_Duration.mean(axis=1)
Module_Reading = data[data['Event_type']=='Reading'].pivot_table(index='User_id', columns='Module', values='Time', aggfunc='count', fill_value=0)
profile['Module_Average_Reading'] = Module_Reading.mean(axis=1)
Module_Activity = data[data['Event_type']=='Reading'].pivot_table(index='User_id', columns='Module', values='Time', aggfunc='count', fill_value=0)
profile['Module_Average_Activity'] = Module_Activity.mean(axis=1)
profile = profile.fillna(0)

Curiosity_list = ['Nunique_Action','Different_Count','Module_Average_Duration','Module_Average_Reading','Module_Average_Activity']

#Calculation of Reactivity

Module_Viewed = data[data['Event_type'] == 'Reading']
Module_Participation = data[data['Event_type'] == 'Activity']
Module_Participation = pd.merge(Module_Viewed, Module_Participation, how='left', on=['User_id', 'Module'])

profile['Module_Participation'] = Module_Participation.groupby('User_id')['Module'].nunique().fillna(0)


def Calculate_Delay(user):
    time_start = user[user['Event_type']=='Reading'].groupby('Module').agg({'Time':'min'})
    time_end = user[user['Event_type']=='Activity'].groupby('Module').agg({'Time':'min'})
    time = (time_end - time_start).iloc[:,0]
    time = time.fillna(pd.Timedelta(days=1))
    time = time.dt.total_seconds()
    return -time.mean()

profile['Module_Average_Delay'] = data.groupby('User_id').apply(Calculate_Delay).fillna(-10000000)


Reactivity_list = ['Rate_Module','Module_Participation','Module_Average_Delay']

#Calculation of Regularity

data['Date'] = data['Time'].dt.date
User_Date_Duration = data.pivot_table(index='User_id', columns='Date', values='Duration', aggfunc='sum', fill_value=0)
time_mean = User_Date_Duration.mean()
profile['User_time_diff'] = User_Date_Duration.apply(lambda x:-sum(abs(x-time_mean)), axis=1)

Regularity_list = ['Days','User_time_diff']

#Calculation of Performance

#read performance data

try:
    performance = pd.read_csv(localpath+'performance.csv')
except FileNotFoundError:
    print("Error: CSV file not found")
    exit()

performance = performance.set_index(performance['User_id'])
performance.drop('User_id',axis= 1, inplace=True)
profile[performance.columns] = performance
profile = profile.fillna(0)

Performance_list = performance.columns.tolist()

profile.to_csv(localpath+'profile.csv', index_label='User_id', columns=profile.columns)
indicators = {
    'Engagement':Engagement_list,
    'Completion': Completion_list,
    'Curiosity' :Curiosity_list,
    'Performance' : Performance_list,
    'Reactivity' : Reactivity_list,
    'Regularity' : Regularity_list}

import json
with open(localpath+'indicators.json', 'w') as f:
    json.dump(indicators, f)

import matplotlib.pyplot as plt

Module_Stats = data.groupby(['Module','User_id']).agg({'Duration': 'mean', 'Event': 'count'})
Module_Stats = Module_Stats.groupby(['Module']).agg({'Duration': 'mean', 'Event': 'mean'})
Module_Stats = Module_Stats.fillna(0)
Module_Stats.columns = ["Time spent","Number of Actions"]
Module_Stats.to_csv(localpath+'Module.csv')
#Module_Stats = Module_Stats.apply(lambda x: (x - min(x)) / (max(x) - min(x))*0.9+0.1)
Module_Stats.plot(kind = 'bar')
plt.savefig(localpath+'Module.png')
#plt.show()

Day_Stats = data.groupby([pd.Grouper(key='Time', freq='D'), 'User_id']).agg({'Duration': 'mean', 'Event': 'count'})
Day_Stats = Day_Stats.groupby('Time').agg({'Duration': 'mean', 'Event': 'mean'})
Day_Stats = Day_Stats.fillna(0)
Day_Stats.columns =  ["Time spent","Number of Actions"]
Day_Stats.plot(kind = 'line')
plt.savefig(localpath+'Days.png')
#plt.show()
