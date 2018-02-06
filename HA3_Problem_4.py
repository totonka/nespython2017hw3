import pandas as pd
import numpy as np

#  Task_1
print('Task1')
df = pd.read_csv('goalies-2014-2016.csv', sep=';')
print(df[['n', 'player', 'season', 'team', 'position',  'games_played']].head())
print('\n\n\n\nTask2')

#Task_2
df['per'] = df.saves / df.shots_against
df['abd'] = df['save_percentage'] - df['per']
abd = max(np.absolute(df['abd']))
print(abd)
print('\n\n\n\nTask3')

#Task_3
print('Mean:')
print(df[['games_played', 'goals_against', 'save_percentage']].mean())
print('\n', 'Standard deviation:')
print(df[['games_played', 'goals_against', 'save_percentage']].std())
print('\n\n\n\nTask4')

#Task_4
df1 = df[df['season'] == '2016-17']
df1 = df1[df1['games_played'] > 40]
df1 = df1[['player', 'save_percentage']]
df1 = df1.sort_values(by='save_percentage', ascending=False)
print(df1.head(1))
print('\n\n\n\nTask5')

#Task_5
dfNew = df[['season', 'player', 'saves']]
mySet = set(dfNew.season)
aa = []
for i in mySet:
    df2 = dfNew[dfNew['season'] == i]
    a = df2.index[df2.saves == df2.saves.max()]
    aa.append(a[0])
dfNew = dfNew.iloc[[aa[0], aa[1], aa[2]]]
dfNew = dfNew.sort_index()
print(dfNew)
print('\n\n\n\nTask6')

#Task_6
dfNew1 = df[['player', 'season', 'wins']][df.wins >= 30]

dfNew16 = dfNew1[['player', 'wins']][dfNew1['season'] == '2016-17']
dfNew16['season'] = pd.Series([3 for i in range(len(dfNew16))])
dfNew16['chk'] = [-3 for i in range(len(dfNew16))]
dfNew15 = dfNew1[['player', 'wins']][dfNew1['season'] == '2015-16']
dfNew15['season'] = [2 for i in range(len(dfNew15))]
dfNew15['chk'] = [-2 for i in range(len(dfNew15))]
dfNew14 = dfNew1[['player', 'wins']][dfNew1['season'] == '2014-15']
dfNew14['season'] = [1 for i in range(len(dfNew14))]
dfNew14['chk'] = [-1 for i in range(len(dfNew14))]
dfNew1 = dfNew16.append([dfNew15, dfNew14])

dfNew1 = dfNew1.sort_values(by='player')
n = 0
bb = []
for i in range(len(dfNew1) - 1):
    if dfNew1.iloc[i]['player'] == dfNew1.iloc[i + 1]['player']:
        n += 1
    elif n == 2:
        bb.append(dfNew1.iloc[i]['player'])
        n = 0
    else:
        n = 0
    if n == 2:
        bb.append(dfNew1.iloc[i]['player'])
        n = 0

dfFin = dfNew1[dfNew1['player'] == bb[0]]
for i in range(1, len(bb)):
    dfFin = dfFin.append(dfNew1[dfNew1['player'] == bb[i]])


dfFin = dfFin.sort_values(by=['chk', 'player'])
dfFin = dfFin.set_index('player')
dfFin = dfFin[['season', 'wins']]
print(dfFin)
