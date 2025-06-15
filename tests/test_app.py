import io
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import streamlit as st

from src import data_processing, eda, model_selection, predictions, database

def test_load_data_csv():
    csv_content = "A,B,C\n1,2,3\n4,5,6"
    fake_file = io.StringIO(csv_content)
    fake_file.name = "test.csv"
    df = data_processing.load_data(fake_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2,3)
    assert "A" in df.columns

def test_load_data_excel():
    # Create a simple DataFrame and save to Excel buffer
    df_original = pd.DataFrame({"A":[1,2],"B":[3,4]})
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_original.to_excel(writer, index=False)
    output.seek(0)
    output.name = "test.xlsx"
    df = data_processing.load_data(output)
    assert isinstance(df, pd.DataFrame)
    assert "A" in df.columns and "B" in df.columns

def test_get_categorical_numeric_columns():
    df = pd.DataFrame({
        "num_col": [1,2,3],
        "cat_col": ["a", "b", "c"]
    })
    cats = data_processing.get_categorical_columns(df)
    nums = data_processing.get_numeric_columns(df)
    assert "cat_col" in cats
    assert "num_col" in nums

def test_eda_perform(capsys):
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["a", "a", "b", "b", "b"]
    })
    # We call perform_eda but check that code runs without errors.
    # For Streamlit usage, ensure it runs without exceptions.
    try:
        eda.perform_eda(df)
    except Exception as e:
        pytest.fail(f"EDA function failed: {e}")

@pytest.mark.parametrize("model_type", ["classification", "regression", "clustering"])
def test_model_selection_valid(model_type):
    from pycaret.datasets import get_data
    if model_type == "classification":
        data = get_data('juice')
        target = 'Purchase'
    elif model_type == "regression":
        data = get_data('boston')
        target = 'medv'
    else:
        data = get_data('jewellery')
        target = None

    model, exp = model_selection.setup_pycaret(data, target, model_type)
    assert model is not None
    assert exp is not None

def test_prediction_functions():
    # Mock model with a predict method returning a known result
    class FakeModel:
        def predict(self, df):
            return [1] * len(df)
    model = FakeModel()
    new_data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    
    preds = predictions.predict_with_model(model, new_data, model_type='clustering')
    assert preds.tolist() == [1, 1]

def test_database_save_and_load(tmp_path):
    # Use a temporary SQLite in-memory database for testing
    from sqlalchemy import create_engine
    test_db_url = "sqlite:///:memory:"
    eng = create_engine(test_db_url)
    
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    # Patch init_engine to use in-memory engine
    with patch('src.database.init_engine') as mock_init_engine:
        mock_init_engine.return_value = (eng, None)
        success = database.save_dataframe(df, "tests", if_exists='replace')
        assert success
        
        loaded_df = database.load_dataframe("tests")
        assert loaded_df is not None
        assert "A" in loaded_df.columns

