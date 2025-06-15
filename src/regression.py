
# src/regression.py
import pandas as pd
from pycaret.regression import create_model, tune_model, finalize_model, predict_model

def improve_regression_model(data: pd.DataFrame, target: str):
    """
    Improve regression model by excluding DummyRegressor and handling fallback.

    Args:
        data: Input DataFrame.
        target: Target column name.

    Returns:
        Tuple (model, experiment) or (None, None) on error.
    """
    try:
        from pycaret.regression import setup, compare_models

        exp = setup(data=data, target=target, html=False)
        best_model = compare_models(exclude=['dummy'])

        if best_model is None:
            raise ValueError("No suitable model found. Please check your data.")

        tuned_model = tune_model(best_model)
        final_model = finalize_model(tuned_model)

        return final_model, exp

    except Exception as e:
        print(f"Error in improving regression model: {str(e)}")
        return None, None
