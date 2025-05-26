import streamlit as st
from pycaret.regression import load_model, predict_model
from model.data_loader import load_regression_data
from model.database import save_prediction
import pandas as pd
import os

def run():
    st.title("📈 Regressão")
    df = load_regression_data()

    st.write("Prévia dos dados:")
    st.dataframe(df.head())

    if not os.path.exists("data/regression_model.pkl"):
        st.warning("Modelo não treinado.")
    else:
        model = load_model("data/regression_model")
        input_text = st.text_input("Digite os valores separados por vírgula (na ordem dos atributos)")
        if input_text:
            try:
                values = [float(v.strip()) for v in input_text.split(',')]
                input_df = pd.DataFrame([values], columns=df.drop(columns=["target"]).columns)
                prediction = predict_model(model, data=input_df)
                pred_value = prediction["prediction_label"][0]
                st.success(f"Predição do valor alvo: **{pred_value:.2f}**")
                save_prediction("regressao", values, pred_value)
            except:
                st.error("Erro ao processar os dados. Verifique o número e tipo dos valores.")
