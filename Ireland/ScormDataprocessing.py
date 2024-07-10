import pandas as pd
import re

# read data
data_scorm = pd.read_csv('scorm_raw.csv')

#del the unused colmnes
old_columns = data_scorm.columns

data_scorm['User_id'] = data_scorm['Moodle user ID']

# define regular expression pattern to extract minutes and seconds
time_pattern = re.compile(r"(\d+\.\d+|\d+)\s*(minute[s]?|second[s]?)")

# define function to convert time string to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return 0.0
    matches = time_pattern.findall(time_str)
    total_seconds = 0.0
    for m in matches:
        if m[1].startswith('min'):
            total_seconds += float(m[0]) * 60
        else:
            total_seconds += float(m[0])
    return total_seconds

# apply the function to the Time column and create a new column called Seconds
data_scorm['Scorm_Duration_1'] = data_scorm['Scorm 1.2.1 time spent'].apply(time_to_seconds)
data_scorm['Scorm_Duration_2']= data_scorm['Scorm 2.2.4 time spent'].apply(time_to_seconds)

data_scorm['Scorm_Result_1'] = data_scorm['Scorm 1.2.1 Status (Grade)'].map({'Passed (1)':1, 'incomplete (0)':0,'no attempt':-1})
data_scorm['Scorm_Result_2'] = data_scorm['Scorm 1.2.1 Status (Grade).1'].map({'Passed (1)':1, 'incomplete (0)':0,'no attempt':-1})

data_scorm.drop(old_columns,axis = 1,inplace =True)

data_scorm.to_csv('performance.csv',index=False)