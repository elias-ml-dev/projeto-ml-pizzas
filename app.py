import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Configuração inicial da página (Título e Ícone na aba do navegador)
st.set_page_config(page_title="Preditor de Pizza", page_icon="🍕")

# --- CARREGAMENTO DOS DADOS ---
# Lendo o arquivo CSV que contém o histórico de diâmetros e preços
df = pd.read_csv("pizzas.csv")

# --- TREINAMENTO DO MODELO ---
# Instanciando o algoritmo de Regressão Linear
modelo = LinearRegression()

# Definindo as variáveis: X (característica/diametro) e Y (alvo/preco)
# O Scikit-Learn espera que a entrada X seja uma tabela (DataFrame)
x = df[["diametro"]]
y = df[["preco"]]

# Treinando o modelo para encontrar a relação matemática entre diâmetro e preço
modelo.fit(x, y)

# --- INTERFACE DO USUÁRIO (FRONT-END) ---
st.title("🍕 Previsão de Preço de Pizza")
st.markdown("Interface desenvolvida para estimar o valor de uma pizza com base no diâmetro.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    # Campo para o usuário interagir e escolher o tamanho da pizza
    st.subheader("Entrada")
    diametro = st.slider("Selecione o diâmetro (cm):", 10, 60, 30)

with col2:
    # Realizando a predição com o valor escolhido no slider
    # [0][0] é usado para extrair o valor numérico puro da matriz de resposta
    preco_previsto = modelo.predict([[diametro]])[0][0]
    
    # Exibindo o resultado da Inteligência Artificial em destaque
    st.subheader("Resultado")
    st.metric("Preço Estimado", f"R$ {preco_previsto:.2f}")

st.divider()


if st.checkbox("Visualizar base de dados original"):
    st.write(df)

# Rodapé simples
st.caption("Desenvolvido para fins de estudo em Machine Learning.")