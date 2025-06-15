# [sem alterações nas importações]
import streamlit as st
import pandas as pd
from datetime import datetime
from src import data_processing, eda, model_selection, predictions, database
import matplotlib.pyplot as plt

from pycaret.classification import plot_model as plot_model_classification, pull as pull_classification
from pycaret.regression import plot_model as plot_model_regression, pull as pull_regression
from pycaret.clustering import plot_model as plot_model_clustering

from src.classification_models import train_classification_model
from src.regression_models import train_regression_models
from src.clustering_models import train_clustering_model

st.set_page_config(page_title="Premium ML App", layout="wide")
st.title("Premium Machine Learning Application")

database.initialize_tables()

session_defaults = {
    'data': None,
    'table_name': "",
    'model_type': None,
    'target': None,
    'selected_features': [],
    'trained_model': None,
    'pycaret_exp': None,
    'prediction': None,
    'r2_score': None
}
for key, val in session_defaults.items():
    st.session_state.setdefault(key, val)


def save_uploaded_dataset(df: pd.DataFrame, default_name: str) -> str:
    st.sidebar.write("### Save Dataset to Database")
    table_name_input = st.sidebar.text_input("Enter dataset table name", value=default_name)
    if st.sidebar.button("Save Dataset"):
        if not table_name_input.strip():
            st.sidebar.error("Please enter a valid table name before saving.")
            return ""
        if database.save_dataframe(df, table_name_input):
            database.save_dataset_metadata(
                name=table_name_input,
                ds_type=st.session_state.get('model_type') or "unknown",
                num_rows=df.shape[0],
                num_cols=df.shape[1],
                description=f"Uploaded dataset saved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                created_at=datetime.now()
            )
            st.sidebar.success(f"Dataset saved successfully as '{table_name_input}'")
            return table_name_input
        else:
            st.sidebar.error(f"Failed to save dataset as '{table_name_input}'")
    return ""


def train_and_save_best_model(df: pd.DataFrame, target: str, model_type: str,
                              selected_features: list, dataset_name: str):
    st.write("### Training and Comparing Models... Please wait.")
    cols = selected_features + ([target] if target else [])
    training_data = df[cols]

    if training_data.isnull().values.any():
        st.error("The dataset contains NaN values. Please clean the data before training.")
        return

    model = None
    exp = None

    try:
        if model_type == 'classification':
            model, exp = train_classification_model(training_data, target)
            results_df = pull_classification()
        elif model_type == 'regression':
            model, results_df = train_regression_models(training_data, target)  # Updated to unpack correctly
            exp = None  # No experiment object for regression in this case
        elif model_type == 'clustering':
            model, exp = train_clustering_model(training_data)
            results_df = None
        else:
            st.error(f"Unknown model type: {model_type}")
            return

        if model is None:
            st.error("Failed to train model.")
            return

        st.session_state['trained_model'] = model
        st.session_state['pycaret_exp'] = exp

        if results_df is not None and not results_df.empty:
            with st.expander("View Full Model Comparison Results"):
                st.dataframe(results_df.style.format(precision=4))

        additional_metrics = {}
        metric_name, metric_value = None, None

        if model_type == 'classification':
            metric_name = 'Accuracy'
            metric_value = results_df.at[results_df.index[0], 'Accuracy'] if 'Accuracy' in results_df else None
            for m in ['AUC', 'Recall', 'Precision', 'F1']:
                if m in results_df:
                    additional_metrics[m] = results_df.at[results_df.index[0], m]

            st.write("### Classification Performance Charts")
            plot_model_classification(model, plot='confusion_matrix', display_format='streamlit')
            plot_model_classification(model, plot='auc', display_format='streamlit')

        elif model_type == 'regression':
            st.write("### Improving Regression Model...")
            final_model = model
            if final_model.__class__.__name__ == 'DummyRegressor':
                st.warning("Best model is DummyRegressor (baseline). No further plots available.")
                return
        
            metric_name = 'R2'
            metric_value = None
            if exp is not None:
                try:
                    df_results = exp.pull()
                    if 'R2' in df_results.columns:
                        metric_value = df_results.loc[0, 'R2']
                except Exception:
                    metric_value = None
        
            st.session_state['r2_score'] = metric_value
        
            for m in ['RMSE', 'MAE', 'MSE']:
                if exp is not None:
                    try:
                        if m in df_results.columns:
                            additional_metrics[m] = df_results.loc[0, m]
                    except Exception:
                        pass
                    
            st.write("### Regression Performance Charts")
            try:
                # Residuals
                fig_residuals = plot_model_regression(final_model, plot='residuals', display_format='matplotlib')
                st.pyplot(fig_residuals)
                plt.clf()
        
                # Prediction error
                fig_pred_error = plot_model_regression(final_model, plot='prediction_error', display_format='matplotlib')
                st.pyplot(fig_pred_error)
                plt.clf()
        
                # Feature importance
                st.subheader("Feature Importance")
                fig_feat_imp = plot_model_regression(final_model, plot='feature', display_format='matplotlib')
                st.pyplot(fig_feat_imp)
                plt.clf()
        
            except Exception as e:
                st.warning(f"Could not render regression plots: {e}")
        
        elif model_type == 'clustering':
            st.write("### Clustering Performance Charts")
            for plot in ['silhouette', 'cluster']:
                plot_model_clustering(model, plot=plot, display_format='streamlit')
                fig = plt.gcf()
                st.pyplot(fig)
                plt.clf()

        model_name = f"{model_type.capitalize()} model on {dataset_name} at {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        database.save_model_metadata(
            name=model_name,
            dataset_name=dataset_name,
            model_type=model_type,
            metric_name=metric_name or 'Unknown',
            metric_value=metric_value or 0.0,
            description=f"Model saved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            created_at=datetime.now()
        )

        st.success(f"Training completed. Model saved as '{model_name}'.")
        st.write("## Best Model Summary")
        st.write(model)

        if metric_name and metric_value:
            st.metric(label=metric_name, value=f"{metric_value:.4f}")
        if additional_metrics:
            cols = st.columns(len(additional_metrics))
            for i, (k, v) in enumerate(additional_metrics.items()):
                cols[i].metric(label=k, value=f"{v:.4f}")

    except Exception as e:
        st.warning(f"Error during training or visualization: {e}")


# Sidebar
st.sidebar.header("Upload or Load Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
load_from_db = st.sidebar.checkbox("Load most recent dataset from database")

if uploaded_file:
    df = data_processing.load_data(uploaded_file)
    default_table_name = uploaded_file.name.replace('.', '_').lower()
    df = data_processing.clean_data(df)  # clean antes de salvar
    st.session_state['data'] = df
    st.session_state['table_name'] = save_uploaded_dataset(df, default_table_name)

elif load_from_db:
    meta = database.get_all_dataset_metadata()
    if meta is not None and not meta.empty:
        latest_name = meta.iloc[0]['name']
        df = database.load_dataframe(latest_name)
        if df is not None:
            df = data_processing.clean_data(df)
            st.session_state['data'] = df
            st.session_state['table_name'] = latest_name
            st.sidebar.success(f"Loaded dataset '{latest_name}' from database")
        else:
            st.sidebar.error("Failed to load dataset")
    else:
        st.sidebar.info("No datasets found in database")

data = st.session_state['data']
if data is not None:
    st.write(f"### Loaded Dataset: {st.session_state['table_name']} ({data.shape[0]} rows, {data.shape[1]} columns)")

    st.header("Select Target and Features")
    columns = data.columns.tolist()
    target = st.selectbox("Select target column (leave blank for clustering)", [""] + columns, key="target_select")
    target = target if target else None
    st.session_state['target'] = target

    features = [col for col in columns if col != target]
    selected_features = st.multiselect("Select feature columns", features, default=features)
    st.session_state['selected_features'] = selected_features

    model_type = st.radio("Select Model Type", ['classification', 'regression', 'clustering'], key="model_radio")
    st.session_state['model_type'] = model_type

    if st.button("Train & Compare Models"):
        if model_type != 'clustering' and not target:
            st.error("Please select a target column.")
        elif not selected_features:
            st.error("Please select at least one feature column.")
        else:
            name = st.session_state['table_name'] or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            train_and_save_best_model(data, target, model_type, selected_features, name)

    if st.checkbox("Explore Dataset (EDA)"):
        eda.perform_eda(data)

    model = st.session_state['trained_model']
    if model is not None:  # Use is not None to check for the model
        st.header("Predict on New Data")
        input_df = predictions.input_new_data_dynamic(data, selected_features, form_key="new_data_form")
        if not input_df.empty:
            preds = predictions.predict_with_model(model, input_df, model_type)
            if preds:
                st.subheader("Prediction Results")
                if model_type == 'classification':
                    st.write(f"**Predicted Class:** {preds.get('predicted_class')}")
                    conf = preds.get('probability')
                    if conf is not None:
                        try:
                            st.write(f"**Confidence:** {float(conf) * 100:.2f}%")
                        except Exception:
                            st.write(f"**Confidence:** {conf}")
                elif model_type == 'regression':
                    st.write(f"**Predicted Value:** {preds.get('predicted_value'):.4f}")
                    r2 = st.session_state.get('r2_score')
                    if r2:
                        st.write(f"**R2 Score:** {r2:.4f}")
                elif model_type == 'clustering':
                    st.write(f"**Predicted Cluster:** {preds.get('predicted_cluster')}")

else:
    st.info("Please upload a dataset or load one from the database to get started.")
