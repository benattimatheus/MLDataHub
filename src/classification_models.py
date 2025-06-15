import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, finalize_model

def train_classification_model(data: pd.DataFrame, target: str, session_id: int = 123):
    """
    Train, tune, and finalize a classification model.

    Args:
        data: Input DataFrame containing features and target.
        target: Target column name.
        session_id: Random seed for reproducibility.

    Returns:
        Tuple of (final_model, experiment_object) or (None, None) on error.
    """
    try:
        exp = setup(data=data, target=target, session_id=session_id, html=False, verbose=False)
        best_model = compare_models()
        
        if best_model.__class__.__name__ == 'DummyClassifier':
            st.warning("Best model is DummyClassifier (baseline). No tuning will be performed.")
            final_model = best_model
        else:
            tuned_model = tune_model(best_model, optimize='Accuracy')
            final_model = finalize_model(tuned_model)
        
        return final_model, exp
    except Exception as e:
        st.error(f"Classification training error: {e}")
        return None, None

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
