import pandas as pd

data = pd.read_csv('goalies-2014-2016.csv', ';')

print(data.iloc[0:5,0:6])

print(abs(data['saves']/data['shots_against']-data['save_percentage']).max())

print(data[['games_played', 'goals_against', 'save_percentage']].mean())
print(data[['games_played', 'goals_against', 'save_percentage']].std())

idx = data[(data['season']=='2016-17')&(data['games_played']>40)]['save_percentage'].idxmax()
print(data.loc[idx,['player','save_percentage']])

ssns = data['season'].unique()
idxs = list()
for ssn in ssns:
    idxs.append(data[data['season']==ssn]['saves'].idxmax())

print(data.loc[idxs,['season', 'player', 'saves']])

tdl = list()

for ssn in ssns:
    td = data[(data['season']==ssn)&(data['wins']>=30)][['player','wins']]
    td = td.set_index('player')
    td.columns = ['wins in season ' + str(ssn)]
    tdl.append(td)


td = tdl[0]
for i in range(1,len(ssns)):
    td = pd.concat([td,tdl[i]], axis = 1)



print(td.dropna())
#for ssn in ssns:
#    d = d[(d['season']==ssn)&(d['wins']<30)].drop()

#print(d['player'])
