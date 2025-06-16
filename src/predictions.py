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
    with st.form(form_key):
        for feature in selected_features:
            col_data = data[feature]

            if pd.api.types.is_numeric_dtype(col_data):
                numeric_col = pd.to_numeric(col_data, errors='coerce').dropna()
                if numeric_col.empty:
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
        new_data_cleaned = new_data.copy()
        for col in new_data_cleaned.columns:
            if pd.api.types.is_numeric_dtype(new_data_cleaned[col]):
                new_data_cleaned[col] = pd.to_numeric(new_data_cleaned[col], errors='coerce')

        if model_type == 'classification':
            # Exibe mapeamento índice → nome da classe (se existir)
            if hasattr(model, "classes_"):
                st.write("Class index to name mapping:")
                for i, cls_name in enumerate(model.classes_):
                    st.write(f"{i} -> {cls_name}")

            from pycaret.classification import predict_model
            pred_df = predict_model(model, data=new_data_cleaned)

            # Pega o valor previsto na coluna 'prediction_label'
            predicted_label_value = pred_df['prediction_label'].values[0]

            # predicted_label_value geralmente já vem string com PyCaret 3.3.2
            predicted_class = predicted_label_value

            result = {'predicted_class': predicted_class}

            # Se a coluna 'prediction_score' existir, retorna a probabilidade
            if 'prediction_score' in pred_df.columns:
                result['probability'] = pred_df['prediction_score'].values[0]

            return result

        elif model_type == 'regression':
            from pycaret.regression import predict_model
            pred_df = predict_model(model, data=new_data_cleaned)
            st.write("Prediction DataFrame:", pred_df)
            st.write("Columns in prediction DataFrame:", pred_df.columns.tolist())  # Debugging line

            label_col = 'Label'  # Ajuste se seu modelo usar outro nome
            if label_col in pred_df.columns:
                return {'predicted_value': pred_df[label_col].values[0]}
            else:
                st.error(f"Expected column '{label_col}' not found in regression results.")
                return {}

        elif model_type == 'clustering':
            pred_cluster = model.predict(new_data_cleaned)
            return {"predicted_cluster": pred_cluster[0] if len(pred_cluster) > 0 else None}

        else:
            st.error(f"Unsupported model type '{model_type}' for prediction.")
            return {}

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {}
