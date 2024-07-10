import pandas as pd
import re

# read data
data_h5p = pd.read_csv('h5P_raw.csv')

data_h5p.rename(columns={'userid': 'User_id'}, inplace=True)

profile_h5p = data_h5p.groupby('User_id').agg({
    'rawscore': 'mean',
    'maxscore': 'mean',
    'duration': 'mean',
    'completion': 'count',
    'success': 'sum'
})

profile_h5p['duration'] = profile_h5p['duration'].apply(lambda x:-x)


profile_h5p.to_csv('performance.csv')