import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')

# Após a análise da base iremos excluir algumas colunas
# Vendas em outros lugares; vendas globais; desenvolvedor
base = base.drop('Other_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('NA_Sales', axis = 1)
base = base.drop('Developer', axis = 1)

# remove as linhas vazias
base = base.dropna(axis = 0)
# remove dados que não são seguros para a rede neural
base = base.loc[base['Global_Sales'] > 1]

# Temos nomes repetidos, por isso iremos remover esses dados
base['Name'].value_counts()
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

# Separando a base de aprendizagem para as bases de resultados
previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
venda_global = base.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

# s 1 0
# r 0 1
from sklearn.compose import ColumnTransformer
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(99,))
camada_oculta1 = Dense(units = 50, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 50, activation = 'sigmoid')(camada_oculta1)
camada_saida = Dense(units = 1, activation = 'linear')(camada_oculta2)


regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida])
regressor.compile(optimizer = 'adam',
                  loss = 'mse')
regressor.fit(previsores, [venda_global],
              epochs = 5000, batch_size = 100)
previsao_global = regressor.predict(previsores)





























