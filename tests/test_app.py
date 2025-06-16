import pytest
import pandas as pd
import numpy as np
from src import data_processing as dp
from src import classification_models as c
from src import regression_models as r
from src import clustering_models as cl

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, 7, 8],
        'B': ['cat', 'dog', 'cat', 'mouse', 'dog', 'cat', 'mouse', 'dog'],
        'Year_Birth': [1990, 1985, 2000, 1995, 1983, 1991, 1988, 1993],
        'High_Card': list('abcdefgh'),
    })

def test_clean_data(sample_df):
    df_clean = dp.clean_data(sample_df.copy())
    assert not df_clean.isnull().values.any()
    for col in df_clean.columns:
        assert pd.api.types.is_numeric_dtype(df_clean[col]) or pd.api.types.is_object_dtype(df_clean[col])

def test_clean_column_names(sample_df):
    df = sample_df.copy()
    df.columns = [' Column 1 ', 'CoL2$', 'Col_3', 'High_Card']  
    df_clean = dp.clean_column_names(df)
    expected = ['column_1', 'col2', 'col_3', 'high_card']  
    assert list(df_clean.columns) == expected

def test_handle_missing_values(sample_df):
    df = sample_df.copy()
    df.loc[0, 'A'] = np.nan

    df_drop = dp.handle_missing_values(df, strategy='drop')
    assert df_drop.isnull().sum().sum() == 0
    assert len(df_drop) < len(df)

    df_mean = dp.handle_missing_values(df, strategy='mean')
    assert df_mean['A'].isnull().sum() == 0

    df_median = dp.handle_missing_values(df, strategy='median')
    assert df_median['A'].isnull().sum() == 0

    df_mode = dp.handle_missing_values(df, strategy='mode')
    assert df_mode.isnull().sum().sum() == 0

    df_ffill = dp.handle_missing_values(df, strategy='ffill')
    df_bfill = dp.handle_missing_values(df, strategy='bfill')
    assert df_ffill.isnull().sum().sum() <= df.isnull().sum().sum()
    assert df_bfill.isnull().sum().sum() <= df.isnull().sum().sum()

def test_reduce_cardinality(sample_df):
    df = sample_df.copy()
    df['B'] = ['cat', 'dog', 'elephant', 'mouse', 'dog', 'cat', 'mouse', 'dog'] 
    df_red = dp.reduce_cardinality(df, max_unique=2)
    assert 'Other' in df_red['B'].values

def test_remove_outliers_iqr(sample_df):
    df = sample_df.copy()
    df_out = dp.remove_outliers_iqr(df, 'A')
    assert df_out['A'].min() >= df['A'].quantile(0.25) - 1.5 * (df['A'].quantile(0.75) - df['A'].quantile(0.25))

def test_create_age_features(sample_df):
    df = sample_df.copy()
    df_new = dp.create_age_features(df)
    age_col = 'Year_Birth_age'
    assert age_col in df_new.columns
    assert all(df_new[age_col] == pd.Timestamp.now().year - df_new['Year_Birth'])

def test_train_regression_models(sample_df):
    df = sample_df.copy()
    df['C'] = [1, 2, 3, 4, 5, 6, 7, 8]
    df = dp.clean_data(df)
    result = r.train_regression_models(df, target='C')
    assert result is not None

def test_train_classification_models(sample_df):
    df = sample_df.copy()
    df['C'] = ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no']  # length matches rows
    result = c.train_classification_model(df, target='C')
    assert result is not None

def test_train_clustering_models(sample_df):
    df = sample_df.copy()
    result = cl.train_clustering_model(df)
    assert result is not None
