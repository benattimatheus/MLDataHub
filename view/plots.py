import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def correlation_heatmap(df):
    st.subheader("Mapa de Calor de Correlação")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

def target_distribution(df, target_column):
    st.subheader("Distribuição da Variável Alvo")
    fig, ax = plt.subplots()
    sns.histplot(df[target_column], kde=True, ax=ax)
    st.pyplot(fig)
