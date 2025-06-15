import pandas as pd
import streamlit as st
from typing import List, Union
import io
import unicodedata
import numpy as np

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file):
    """Load data from a CSV or Excel file."""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by handling missing values and encoding categorical variables."""
    # Handle missing values (example: fill with mode for categorical, mean for numerical)
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)  # Fill categorical with mode
        else:
            df[column].fillna(df[column].mean(), inplace=True)  # Fill numerical with mean

    # Encode categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        st.warning("DataFrame is empty. No column names to clean.")
        return df

    def normalize_column(col):
        col = col.strip().lower()
        col = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('utf-8')
        col = col.replace(' ', '_')
        col = ''.join(c for c in col if c.isalnum() or c == '_')
        return col

    df.columns = [normalize_column(col) for col in df.columns]
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    df = df.copy()
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif strategy == "median":
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    elif strategy == "mode":
        for col in df.columns:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
        return df
    elif strategy == "ffill":
        return df.fillna(method='ffill')
    elif strategy == "bfill":
        return df.fillna(method='bfill')
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")

def reduce_cardinality(df: pd.DataFrame, max_unique: int = 25) -> pd.DataFrame:
    """
    Reduce cardinality of categorical columns that have unique values above the limit.
    """
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].nunique() > max_unique:
            top = df[col].value_counts().nlargest(max_unique).index
            df[col] = df[col].apply(lambda x: x if x in top else 'Other')
    return df

def remove_outliers_iqr(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Remove outliers from the target using the IQR method.
    """
    if target not in df.columns:
        return df
    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[target] >= lower) & (df[target] <= upper)]

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age features from columns with 'year' in their name.
    """
    year_cols = [col for col in df.columns if 'year' in col.lower() and pd.api.types.is_numeric_dtype(df[col])]
    for col in year_cols:
        df[f'{col}_age'] = pd.Timestamp.now().year - df[col]
    return df


import streamlit as st
import pandas as pd
from pycaret.clustering import setup, create_model, assign_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Setup PyCaret
        exp = setup(data=pd.DataFrame(scaled_data, columns=data.columns), html=False, verbose=False, session_id=session_id)
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

def determine_optimal_clusters(data: pd.DataFrame, max_k: int = 10):
    """
    Determine the optimal number of clusters using the Elbow Method.

    Args:
        data: Input DataFrame for clustering.
        max_k: Maximum number of clusters to test.

    Returns:
        None (plots the elbow curve in Streamlit).
    """
    inertia = []
    for k in range(2, max_k + 1):  # start at 2 because num_clusters must be > 1
        model = create_model('kmeans', num_clusters=k)
        inertia.append(model.inertia_)

    # Plot the Elbow curve with matching x-axis range
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), inertia, marker='o')  # Fix: range matches inertia length
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid()
    st.pyplot(plt.gcf())
    plt.clf()
