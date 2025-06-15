import streamlit as st
import pandas as pd
from typing import Dict, List

import streamlit as st
import pandas as pd
from typing import List

import streamlit as st
import pandas as pd
from typing import List

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

    return pd.DataFrame()  # empty when not submitted


def predict_with_model(model, new_data: pd.DataFrame, model_type: str):
    """
    Return prediction results with confidence or probabilities where appropriate.
    """

    if new_data.empty:
        st.warning("No input data provided for prediction.")
        return None

    try:
        # Clean new_data numeric columns before prediction
        new_data_cleaned = new_data.copy()
        for col in new_data_cleaned.columns:
            if pd.api.types.is_numeric_dtype(new_data_cleaned[col]):
                new_data_cleaned[col] = pd.to_numeric(new_data_cleaned[col], errors='coerce')

        if model_type == 'classification':
            from pycaret.classification import predict_model
            pred_df = predict_model(model, data=new_data_cleaned)
            # [rest of your code unchanged...]

        elif model_type == 'regression':
            from pycaret.regression import predict_model
            pred_df = predict_model(model, data=new_data_cleaned)
            # [rest of your code unchanged...]

        elif model_type == 'clustering':
            pred_cluster = model.predict(new_data_cleaned)[0]
            return {"predicted_cluster": pred_cluster}

        else:
            st.error(f"Unsupported model type {model_type} for prediction.")
            return None

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
