import streamlit as st
import pandas as pd
from pycaret.clustering import setup, create_model, assign_model, get_config
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
from pycaret.clustering import setup, create_model, assign_model
from sklearn.preprocessing import StandardScaler

def train_clustering_model(data: pd.DataFrame, model_name='kmeans', num_clusters=3, session_id: int = 123):
    try:
        print(data.dtypes)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        exp = setup(data=pd.DataFrame(scaled_data, columns=data.columns), html=False, verbose=False, session_id=session_id)
        model = create_model(model_name, num_clusters=num_clusters)
        clustered_data = assign_model(model)
        return model, exp, clustered_data

    except Exception as e:
        st.error(f"Clustering training error: {e}")
        return None, None, None


def determine_optimal_clusters(data: pd.DataFrame, max_k: int = 10):
    """
    Determine optimal number of clusters via the Elbow Method.
    """
    from sklearn.cluster import KMeans

    inertia = []
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=123)
        model.fit(scaled_data)
        inertia.append(model.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), inertia, marker='o')
    plt.title('Elbow Method - Optimal Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()

def plot_cluster_summary(clustered_data: pd.DataFrame):
    """
    Display cluster summary: cluster sizes and feature means.
    """
    st.subheader("📊 Cluster Sizes")
    st.write(clustered_data['Cluster'].value_counts().rename("Count").to_frame())

    st.subheader("📈 Average values by Cluster")
    mean_df = clustered_data.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(mean_df.style.format(precision=2))

    if clustered_data.shape[1] >= 3:
        st.subheader("📊 Cluster Distribution (First 2 Features)")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=clustered_data.iloc[:, 0],
            y=clustered_data.iloc[:, 1],
            hue=clustered_data['Cluster'],
            palette='Set2',
            ax=ax
        )
        plt.xlabel(clustered_data.columns[0])
        plt.ylabel(clustered_data.columns[1])
        plt.title("Clusters (by first 2 features)")
        st.pyplot(fig)
        plt.clf()
