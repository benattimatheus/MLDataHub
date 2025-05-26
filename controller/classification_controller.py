import streamlit as st
from pycaret.classification import load_model, predict_model
from model.data_loader import load_classification_data
from model.database import save_prediction
import pandas as pd
import os

def run():
    st.title("🔍 Classificação")
    df = load_classification_data()

    st.write("Prévia dos dados:")
    st.dataframe(df.head())

    if not os.path.exists("data/classification_model.pkl"):
        st.warning("Modelo não treinado.")
    else:
        model = load_model("data/classification_model")
        inputs = st.text_input("Digite os valores separados por vírgula (na ordem dos atributos)")
        if inputs:
            values = [float(v.strip()) for v in inputs.split(',')]
            input_df = pd.DataFrame([values], columns=df.drop(columns=["target"]).columns)
            prediction = predict_model(model, data=input_df)
            pred_value = prediction["prediction_label"][0]
            st.success(f"Predição: {pred_value}")
            save_prediction("classificacao", values, pred_value)
