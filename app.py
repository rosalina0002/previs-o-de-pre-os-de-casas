
import streamlit as st
import pandas as pd
import joblib

st.title("Previsão do Preço de Casas")

# Entradas do usuário
area = st.slider("Área do Lote (LotArea)", 1000, 20000, 8000)
ano_venda = st.slider("Ano de Venda (YrSold)", 2006, 2010, 2008)
mes_venda = st.slider("Mês de Venda (MoSold)", 1, 12, 6)
valor_misc = st.slider("Valor Extra (MiscVal)", 0, 10000, 0)
enclosed_porch = st.slider("Enclosed Porch", 0, 500, 0)

# Dados de entrada
X = pd.DataFrame([[area, ano_venda, mes_venda, valor_misc, enclosed_porch]],
                 columns=["LotArea", "YrSold", "MoSold", "MiscVal", "EnclosedPorch"])

# Carregar modelo e scaler reais
modelo = joblib.load("modelo_lr.pkl")
scaler = joblib.load("scaler.pkl")

# Normalizar e prever
X_norm = scaler.transform(X)
preco_previsto = modelo.predict(X_norm)[0]

# Resultado
st.subheader("Resultado")
st.write(f"Preço Previsto: {preco_previsto:,.2f} Kz")

st.caption("Este app utiliza um modelo treinado para prever preços de casas.")
