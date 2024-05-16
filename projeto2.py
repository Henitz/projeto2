from pipes import quote
from urllib.parse import quote_plus

import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import streamlit as st
import warnings

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# URL do arquivo CSV no GitHub (com espaços substituídos por +)
base_url = 'https://raw.githubusercontent.com/Henitz/projeto2/master/'
file_name = 'Dados Históricos - Petróleo Brent Futuros (8).csv'
encoded_file_name = quote_plus(file_name)
csv_url = f'{base_url}{encoded_file_name}'


# Carregar dados do Brent
df = pd.read_csv(csv_url)

# Carregar dados do Brent
# df = pd.read_csv('Dados Históricos - Petróleo Brent Futuros (8).csv')

# Renomear colunas
df = df.rename(columns={'Data': 'ds', 'Último': 'y'})

# Substituir vírgulas por pontos na coluna 'y' e converter para numérico
df['y'] = df['y'].str.replace(',', '.').astype(float)

df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y')

# Remover colunas desnecessárias
colunas_para_remover = ['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
df = df.drop(columns=colunas_para_remover)

# Obter feriados do Reino Unido de 1970 a 2025
uk_holidays = holidays.UK(years=range(1970, 2026))
holiday_dates = list(uk_holidays.keys())

# Criar DataFrame de feriados
feriados_uk = pd.DataFrame({
    'holiday': 'feriados_uk',
    'ds': pd.to_datetime(holiday_dates),
    'lower_window': 0,
    'upper_window': 1,
})


# Função para prever usando Prophet
def prevendo(df, data, flag):
    m = Prophet(holidays=feriados_uk)
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    if flag:
        result = forecast[forecast['ds'] == data]
        if not result.empty:
            return result['yhat'].values[0]
        return None
    else:
        return m, forecast


# Interface com Streamlit
st.title("Previsão de Preços do Petróleo Brent - 16/05/2024 a 16/05/2025")

data_selecionada = st.date_input("Selecione uma data")

if st.button('Prever'):
    data_formatada = pd.to_datetime(data_selecionada).strftime('%Y-%m-%d')
    previsao = prevendo(df, data_formatada, True)

    if previsao is None:
        st.write(f"A data {data_formatada} não está disponível nas previsões ou é feriado/final de semana.")
    else:
        st.write(f"Valor previsto para {data_formatada}: {previsao:.2f}")

# Gráficos de Previsão
st.write("### Gráfico de Previsão")
model, forecast = prevendo(df, datetime.now().strftime('%Y-%m-%d'), False)

if forecast is not None:
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.write("### Gráfico de Componentes da Previsão")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Filtrar y_true para corresponder às datas em forecast
    df_filtered = df[df['ds'].isin(forecast['ds'])]
    y_true = df_filtered['y'].values
    y_pred = forecast.loc[forecast['ds'].isin(df_filtered['ds']), 'yhat'].values

    # Métricas de Avaliação
    st.write("### Avaliação do Modelo")

    st.write("""
    - **MAE (Mean Absolute Error):** Mede a média dos erros absolutos entre as previsões e os valores reais. Quanto menor o valor, melhor o modelo.
    - **MSE (Mean Squared Error):** Mede a média dos erros quadrados entre as previsões e os valores reais. Dá mais peso aos grandes erros. Quanto menor o valor, melhor o modelo.
    - **RMSE (Root Mean Squared Error):** É a raiz quadrada do MSE. Mantém as unidades dos dados originais e é interpretado da mesma forma que o MSE.
    - **MAPE (Mean Absolute Percentage Error):** Mede a média dos erros percentuais absolutos entre as previsões e os valores reais. Quanto menor o valor, melhor o modelo.
    """)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}")
else:
    st.write("As previsões ainda não estão disponíveis.")
