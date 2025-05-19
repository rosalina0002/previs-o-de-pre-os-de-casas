# Treinamento de Modelo para Previsão de Preços de Casas

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

print("✅ Iniciando o treinamento...")

# Carregar dados (certifique-se de que 'train.csv' está no mesmo diretório)
dados = pd.read_csv('train.csv')

# Selecionar variáveis do modelo
colunas_modelo = ['LotArea', 'YrSold', 'MoSold', 'MiscVal', 'EnclosedPorch']
X = dados[colunas_modelo]
y = dados['SalePrice']

# Tratar valores ausentes
X = X.fillna(0)
y = y.fillna(y.mean())

# Normalizar os dados
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Treinar modelo
modelo = LinearRegression()
modelo.fit(X_norm, y)

# Salvar modelo e scaler
joblib.dump(modelo, 'modelo_lr.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Modelo treinado e arquivos salvos com sucesso!")
