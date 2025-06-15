import streamlit as st
import pandas as pd
from pycaret.clustering import setup, create_model, assign_model

def train_clustering_model(data: pd.DataFrame, session_id: int = 123):
    """
    Train a clustering model and return the model and experiment object.

    Args:
        data: Input DataFrame for clustering.
        session_id: Random seed for reproducibility.

    Returns:
        Tuple of (model, experiment_object) or (None, None) on error.
    """
    try:
        exp = setup(data=data, html=False, verbose=False, session_id=session_id)
        model = create_model('kmeans')
        return model, exp
    except Exception as e:
        st.error(f"Clustering training error: {e}")
        return None, None

def predict_clustering(model, data: pd.DataFrame):
    """
    Assign clusters to the data using the trained clustering model.

    Args:
        model: Trained clustering model.
        data: DataFrame to assign clusters.

    Returns:
        DataFrame with cluster assignments.
    """
    try:
        clustered_data = assign_model(model, data)
        return clustered_data
    except Exception as e:
        st.error(f"Clustering prediction error: {e}")
        return None
