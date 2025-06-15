import streamlit as st
import pandas as pd
from typing import Optional, Tuple, Any, List

def setup_pycaret(
    data: pd.DataFrame,
    target: Optional[str],
    model_type: str,
    categorical_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    session_id: int = 123
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Setup and run a PyCaret experiment.

    Args:
        data: Input DataFrame.
        target: Target column name (ignored for clustering).
        model_type: One of 'classification', 'regression', 'clustering'.
        categorical_features: Optional list of categorical features.
        numeric_features: Optional list of numeric features.
        session_id: Random seed.

    Returns:
        Tuple (best_model, experiment_object), or (None, None) on error.
    """
    try:
        setup_kwargs = {
            'data': data,
            'session_id': session_id,
            'verbose': False,
            'html': False,
            'use_gpu': False
        }

        if model_type in ['classification', 'regression']:
            if target is None:
                st.error("Target column must be specified for classification or regression.")
                return None, None
            setup_kwargs['target'] = target
            if categorical_features:
                setup_kwargs['categorical_features'] = categorical_features
            if numeric_features:
                setup_kwargs['numeric_features'] = numeric_features

        if model_type == 'classification':
            from pycaret.classification import setup, compare_models
            exp = setup(**setup_kwargs)
            best_model = compare_models()

        elif model_type == 'regression':
            from pycaret.regression import setup, compare_models
            exp = setup(**setup_kwargs)
            best_model = compare_models()

        elif model_type == 'clustering':
            from pycaret.clustering import setup, create_model
            exp = setup(**setup_kwargs)
            best_model = create_model('kmeans')

        else:
            st.error("Invalid model type. Choose from 'classification', 'regression', or 'clustering'.")
            return None, None

        st.success(f"Best {model_type} model: {best_model}")
        return best_model, exp

    except Exception as e:
        st.error(f"Error during PyCaret setup or training: {str(e)}")
        return None, None
