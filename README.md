# previs-o-de-pre-os-de-casas

# Previsão de Preço de Casas 🏠

Este projeto utiliza Machine Learning para prever o preço de casas com base em dados de entrada fornecidos pelo usuário. A aplicação foi desenvolvida com **Streamlit**, permitindo uma interface web simples e interativa.

## 📌 Funcionalidades

- Interface interativa para inserir dados relevantes de uma propriedade:
  - Área do lote (`LotArea`)
  - Ano e mês da venda (`YrSold`, `MoSold`)
  - Valor adicional (`MiscVal`)
  - Área de varanda fechada (`EnclosedPorch`)
- Carregamento de modelo e scaler previamente treinados
- Normalização dos dados de entrada
- Previsão do preço da casa
- Exibição do resultado formatado com moeda brasileira

## 🧠 Modelagem

O modelo foi treinado previamente utilizando algoritmos de regressão linear e normalização dos dados. A aplicação carrega:
- `modelo_lr.pkl`: Modelo de regressão serializado com `joblib`
- `scaler.pkl`: Scaler de normalização dos dados

A modelagem e validação estão descritas nos notebooks:
- `Notebook_Projeto_Completo.ipynb`
- `Notebook_Validacao_Cruzada.ipynb`

## ▶️ Como Executar

1. Instale os pacotes necessários:
   ```bash
   pip install -r requirements.txt
