import streamlit as st
import pandas as pd
from typing import List, Dict, Any

def input_new_data_dynamic(data: pd.DataFrame, selected_features: List[str], form_key: str) -> pd.DataFrame:
    """
    Dynamically build input form with ranges/dropdowns according to data feature types and unique values.
    All numeric inputs use integer sliders (floats coerced to int for slider min/max).
    """
    st.subheader("Input New Data For Prediction")

    input_data = {}
    with st.form(form_key):  # Use the unique form key passed as an argument
        for feature in selected_features:
            col_data = data[feature]

            if pd.api.types.is_numeric_dtype(col_data):
                numeric_col = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_col) == 0:
                    st.error(f"Feature '{feature}' has no valid numeric data.")
                    continue
                min_val = int(numeric_col.min())
                max_val = int(numeric_col.max())
                val = st.slider(f"{feature}", min_value=min_val, max_value=max_val, step=1, format="%d")
            else:
                options = sorted(col_data.dropna().unique())
                val = st.selectbox(f"{feature}", options)

            input_data[feature] = val

        submitted = st.form_submit_button("Predict")

    if submitted:
        df_new = pd.DataFrame([input_data])
        st.write("Input data:")
        st.dataframe(df_new)
        return df_new

    return pd.DataFrame()  # Return empty DataFrame when not submitted


def predict_with_model(model: Any, new_data: pd.DataFrame, model_type: str) -> Dict[str, Any]:
    """
    Return prediction results with confidence or probabilities where appropriate.
    """
    if new_data.empty:
        st.warning("No input data provided for prediction.")
        return {}

    try:
        # Clean new_data numeric columns before prediction
        new_data_cleaned = new_data.copy()
        for col in new_data_cleaned.columns:
            if pd.api.types.is_numeric_dtype(new_data_cleaned[col]):
                new_data_cleaned[col] = pd.to_numeric(new_data_cleaned[col], errors='coerce')

        if model_type == 'classification':
            from pycaret.classification import predict_model
            pred_df = predict_model(model, data=new_data_cleaned)
        
            # Verifica quais colunas existem
            if 'Label' in pred_df.columns and 'Score' in pred_df.columns:
                return {
                    'predicted_class': pred_df['Label'],
                    'probability': pred_df['Score']
                }
            elif 'prediction_label' in pred_df.columns and 'prediction_score' in pred_df.columns:
                return {
                    'predicted_class': pred_df['prediction_label'],
                    'probability': pred_df['prediction_score']
                }
            else:
                st.error("Nenhuma das colunas esperadas ('Label', 'Score' ou 'prediction_label', 'prediction_score') foi encontrada nos resultados.")
                st.dataframe(pred_df)  # Mostra o DataFrame completo para depuração
                return {}
        
        elif model_type == 'regression':
            from pycaret.regression import predict_model
            pred_df = predict_model(model, data=new_data_cleaned)
            st.write("Prediction DataFrame:", pred_df)  # Debugging line to check the structure

            # Check the actual column names in the returned DataFrame
            st.write("Columns in prediction DataFrame:", pred_df.columns.tolist())  # List the columns

            if 'Label' in pred_df.columns:
                return {
                    'predicted_value': pred_df['Label']  # Access the predicted value
                }
            else:
                st.error("Expected column 'Label' not found in regression results.")
                return {}

        elif model_type == 'clustering':
            pred_cluster = model.predict(new_data_cleaned)
            return {"predicted_cluster": pred_cluster}

        else:
            st.error(f"Unsupported model type {model_type} for prediction.")
            return {}

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {}

def predict_classification(model, input_data: pd.DataFrame):
    """
    Make predictions using the trained classification model.

    Args:
        model: Trained classification model.
        input_data: DataFrame containing input features for prediction.

    Returns:
        Predictions as a Series.
    """
    predictions = model.predict(input_data)
    return predictions
