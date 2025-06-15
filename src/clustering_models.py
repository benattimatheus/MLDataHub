# clustering_models.py
import streamlit as st
import pandas as pd
from pycaret.clustering import setup, create_model, assign_model

def train_clustering_model(data: pd.DataFrame, session_id: int = 123):
    try:
        exp = setup(
            data=data,
            session_id=session_id,
            html=False,
            verbose=False
        )
        model = create_model('kmeans')
        return model, exp
    except Exception as e:
        st.error(f"Clustering training error: {e}")
        return None, None

def predict_clustering(model, data: pd.DataFrame):
    try:
        clustered_data = assign_model(model, data)
        return clustered_data
    except Exception as e:
        st.error(f"Clustering prediction error: {e}")
        return None

# clustering_workflow.py (new or inside clustering_models.py)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pycaret.clustering import setup, create_model, assign_model

def train_and_plot_clustering_model(data: pd.DataFrame, session_id: int = 123):
    try:
        # Drop non-feature columns
        if 'CustomerID' in data.columns:
            data = data.drop(columns=['CustomerID'])

        # Preprocess: encode non-numeric
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            st.warning(f"Encoding non-numeric columns for clustering: {non_numeric_cols}")
            data = pd.get_dummies(data, columns=non_numeric_cols, drop_first=True)

        # Check for remaining non-numeric columns
        remaining_non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        if remaining_non_numeric:
            st.error(f"Non-numeric columns remain after encoding: {remaining_non_numeric}")
            return None, None

        if data.isnull().values.any():
            st.error("Clustering data contains NaNs; clean data before training.")
            return None, None

        # Setup and train model
        exp = setup(data=data, session_id=session_id, html=False, verbose=False)
        model = create_model('kmeans')

        # Predict clusters
        clustered_data = assign_model(model, data)

        # Plot clusters
        _plot_clusters(clustered_data)

        return model, exp

    except Exception as e:
        st.error(f"Clustering training and plotting error: {e}")
        return None, None


def _plot_clusters(clustered_data: pd.DataFrame):
    cluster_column = 'Cluster'  # default column added by assign_model
    pca = PCA(n_components=2)
    features = clustered_data.drop(columns=[cluster_column])
    pca_result = pca.fit_transform(features)

    plot_data = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    plot_data[cluster_column] = clustered_data[cluster_column].values

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x='PCA1', y='PCA2', hue=cluster_column, palette='viridis', s=100)
    plt.title('Clusters visualized with PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title=cluster_column)
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.clf()
