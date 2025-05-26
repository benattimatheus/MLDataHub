import streamlit as st
from pycaret.clustering import load_model, assign_model
from model.data_loader import load_clustering_data
from model.database import save_prediction
import pandas as pd
import os

def run():
    st.title("🧪 Clusterização")
    df = load_clustering_data()

    st.write("Prévia dos dados:")
    st.dataframe(df.head())

    if not os.path.exists("data/clustering_model.pkl"):
        st.warning("Modelo não treinado.")
    else:
        model = load_model("data/clustering_model")
        df_clustered = assign_model(model, data=df)
        st.write("Dados com clusters atribuídos:")
        st.dataframe(df_clustered.head())

        st.bar_chart(df_clustered["Cluster"].value_counts())
