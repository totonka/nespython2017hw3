#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np




#1
df = pd.DataFrame()
df = pd.read_csv('./goalies-2014-2016.csv', sep=';')
df.iloc[0:4,0:6]

#2
df1=df
df1['sp']=df1['saves']/df1['shots_against']
df1['sp']=df1['sp'].round(3)
df1[['sp','save_percentage']]
max(abs(df1['sp'].fillna(0)-df1['save_percentage']))

#3
df2=df1
print('games_played',df2['games_played'].mean(),'\n',
      'goals_against',df2['goals_against'].mean(),'\n',
      'save_percentage',df2['save_percentage'].mean())

print('games_played',df2['games_played'].std(),'\n',
      'goals_against',df2['goals_against'].std(),'\n',
      'save_percentage',df2['save_percentage'].std())

#4
df2[(df['season']=='2016-17') & (df1['games_played']>40)][['player',
    'save_percentage']].sort_values(by=['save_percentage'],ascending=False)[:1]

#5
df3=df2
#df3.groupby(['season'], sort=False)['saves'].max()
df3[df3.groupby(['season'])['saves'].transform(max) == df3['saves']][['season',
    'player','saves']]

#6
df4=df3
df4.groupby(['player','season'])['wins'].transform(max)
grouped = df4[(df4.groupby(['player','season'])['wins'].transform(max) == 
               df4['wins']) &(df4['wins']>30)][['player','season','wins']]

grouped.groupby('player')['season'].count()>=3 #<- True is the answer

df4[df4['wins']>30][['player','season','wins']].groupby('player')['season'].count()>=3
#True is the answer


df5=df4.set_index('player')
df6=df5[['season','wins']]
df7=df6[df5['wins']>30]
(df7.groupby('player')['season'].count()>=3)