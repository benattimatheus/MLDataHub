import streamlit as st
from controller.classification_controller import run as run_classification
from controller.regression_controller import run as run_regression
from controller.clustering_controller import run as run_clustering
from model.database import init_db

def run():
    st.sidebar.title("📊 Escolha o tipo de problema")
    page = st.sidebar.radio("Tipo", ["Classificação", "Regressão", "Clusterização"])

    init_db()

    if page == "Classificação":
        run_classification()
    elif page == "Regressão":
        run_regression()
    elif page == "Clusterização":
        run_clustering()
