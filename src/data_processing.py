import pandas as pd
import streamlit as st
import unicodedata

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        raise
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
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
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].nunique() > max_unique:
            top = df[col].value_counts().nlargest(max_unique).index
            df[col] = df[col].apply(lambda x: x if x in top else 'Other')
    return df

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    year_cols = [col for col in df.columns if 'year' in col.lower() and pd.api.types.is_numeric_dtype(df[col])]
    for col in year_cols:
        df[f'{col}_age'] = pd.Timestamp.now().year - df[col]
    return df

def remove_outliers_iqr(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        return df
    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[target] >= lower) & (df[target] <= upper)]

def clean_data(df: pd.DataFrame, missing_value_strategy: str = "drop", target: str = None) -> pd.DataFrame:
    df = clean_column_names(df)
    df = handle_missing_values(df, missing_value_strategy)
    df = reduce_cardinality(df)
    df = create_age_features(df)
    if target:
        df = remove_outliers_iqr(df, target)
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def preprocess_for_clustering(df: pd.DataFrame, missing_value_strategy='drop', target=None) -> pd.DataFrame:
    df = clean_data(df, missing_value_strategy=missing_value_strategy, target=target)
    df = encode_categorical(df)

    # Ensure all columns are numeric after encoding
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        st.error(f"Non-numeric columns detected after encoding: {non_numeric_cols}.")
        raise ValueError("DataFrame contains non-numeric columns after preprocessing.")

    return df
