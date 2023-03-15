"""
Author: Lucas Henrique Alves Rosa
Código: fazer a regresão de uma base de dados para um valor que é o preço de automóveis usados com base em suas especificações
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

# Removendo colunas que não fazem sentido para a rede neural
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
# Alguns dados possuem uma grande ou pouca variabilidade por isso serão removidos
# base['name'].value_counts()
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)

# Verificação dos preços do automovel, não é interessante valores como 0 e 10 euros
# base.price.mean()
base = base[base.price > 10]
# Valores muito altos também demonstram incosistencias
base = base.loc[base.price < 350000]

# Verificação se existem valores vazios e vamos substituir pela que mais se repete
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts() # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() # benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() # nein

# para preencher os vazios
valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value = valores)

# Separando a base de dados
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Transforma os dados categoricos em numericos
labelencoder_previsores = LabelEncoder()
previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:, 10])

from sklearn.compose import ColumnTransformer
# Fazer o dummy nas classificações
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough') 
previsores = onehotencoder.fit_transform(previsores).toarray()

# Criando a rede neural
regressor = Sequential()
"""
Parametros:
    units: neuronios (entrada+saida)/2
"""
regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
regressor.add(Dense(units = 158, activation = 'relu'))
# Camada de saida
"""
Parametros:
    activation: Função de ativação
        linear: Não faz nada
"""
regressor.add(Dense(units = 1, activation = 'linear'))
"""
Parametros:
    loss: Função de perda
        mean_absolute_error: média de Erro absoluto
"""
regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, batch_size = 300, epochs = 100)

previsoes = regressor.predict(previsores)
preco_real.mean()
previsoes.mean()
