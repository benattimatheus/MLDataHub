import pandas as pd
import streamlit as st
from typing import List, Union
import io

def load_data(uploaded_file: Union[io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """
    Load CSV or Excel file into a DataFrame.

    Args:
        uploaded_file: Uploaded file-like object.

    Returns:
        DataFrame with the data, or raises ValueError if format unsupported.
    """
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        raise
    return df

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of numeric columns in the DataFrame.
    """
    return df.select_dtypes(include=['int64', 'float64']).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of categorical/object columns in the DataFrame.
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize column names: lowercase, no spaces, replace special characters.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values with a specified strategy.

    Args:
        df: Input DataFrame.
        strategy: One of ['drop', 'mean', 'median'].

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
