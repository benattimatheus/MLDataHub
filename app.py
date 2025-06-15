import streamlit as st
import pandas as pd
from datetime import datetime
from src import data_processing, eda, model_selection, predictions, database
import matplotlib.pyplot as plt
from pycaret.classification import plot_model as plot_model_classification
from pycaret.regression import plot_model as plot_model_regression
from pycaret.clustering import plot_model as plot_model_clustering

st.set_page_config(page_title="Premium ML App", layout="wide")
st.title("Premium Machine Learning Application")

# Initialize database tables once on app start
database.initialize_tables()

# Session state defaults
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'table_name' not in st.session_state:
    st.session_state['table_name'] = ""
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = None
if 'target' not in st.session_state:
    st.session_state['target'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'pycaret_exp' not in st.session_state:
    st.session_state['pycaret_exp'] = None
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None


def save_uploaded_dataset(df: pd.DataFrame, default_name: str) -> str:
    st.sidebar.write("### Save Dataset to Database")
    table_name_input = st.sidebar.text_input("Enter dataset table name", value=default_name)
    if st.sidebar.button("Save Dataset"):
        if table_name_input.strip() == "":
            st.sidebar.error("Please enter a valid table name before saving.")
            return ""
        if database.save_dataframe(df, table_name_input):
            database.save_dataset_metadata(
                name=table_name_input,
                ds_type=st.session_state.get('model_type') or "unknown",
                num_rows=df.shape[0],
                num_cols=df.shape[1],
                description=f"Uploaded dataset saved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                created_at=datetime.now(),
            )
            st.sidebar.success(f"Dataset saved successfully as '{table_name_input}'")
            return table_name_input
        else:
            st.sidebar.error(f"Failed to save dataset as '{table_name_input}'")
            return ""
    return ""


def train_and_save_best_model(df: pd.DataFrame, target: str, model_type: str,
                              selected_features: list, dataset_name: str):
    st.write("### Training and Comparing Models... Please wait.")

    cols = selected_features.copy()
    if target:
        cols.append(target)
    training_data = df[cols]

    # Check for NaN values
    if training_data.isnull().sum().any():
        st.error("The dataset contains NaN values. Please clean the data before training.")
        return

    model, exp = model_selection.setup_pycaret(training_data, target, model_type)
    if model is None:
        st.error("Failed to train model, see logs.")
        return

    st.session_state['trained_model'] = model
    st.session_state['pycaret_exp'] = exp

    try:
        results_df = exp.pull()
        with st.expander("View Full Model Comparison Results"):
            st.dataframe(results_df.style.format(precision=4))

        metric_name, metric_value = None, None
        additional_metrics = {}

        if model_type == 'classification':
            metric_name = 'Accuracy'
            metric_value = results_df.loc[results_df.index[0], 'Accuracy'] if 'Accuracy' in results_df.columns else None
            for metric in ['AUC', 'Recall', 'Precision', 'F1']:
                if metric in results_df.columns:
                    additional_metrics[metric] = results_df.loc[results_df.index[0], metric]

            st.write("### Classification Performance Charts")
            plot_model_classification(model, plot='confusion_matrix', display_format='streamlit')
            plot_model_classification(model, plot='auc', display_format='streamlit')

        elif model_type == 'regression':
            metric_name = 'R2'
            metric_value = results_df.loc[results_df.index[0], 'R2'] if 'R2' in results_df.columns else None
            st.session_state['r2_score'] = metric_value
            for metric in ['RMSE', 'MAE', 'MSE']:
                if metric in results_df.columns:
                    additional_metrics[metric] = results_df.loc[results_df.index[0], metric]

            st.write("### Regression Performance Charts")

            # Attempt to plot residuals
            try:
                plot_model_regression(model, plot='residuals', display_format='streamlit')
                fig2 = plt.gcf()
                if fig2.get_axes():  # Check if there are any axes in the figure
                    st.pyplot(fig2)
                else:
                    st.warning("The 'Residuals' plot is empty.")
                plt.clf()
            except Exception as e:
                st.warning(f"Could not render 'residuals' plot: {e}")

        elif model_type == 'clustering':
            metric_name = 'Silhouette'
            metric_value = results_df.loc[results_df.index[0], 'Silhouette'] if 'Silhouette' in results_df.columns else None

            st.write("### Clustering Performance Charts")
            plot_model_clustering(model, plot='silhouette', display_format='streamlit')
            fig1 = plt.gcf()
            st.pyplot(fig1)
            plt.clf()

            plot_model_clustering(model, plot='cluster', display_format='streamlit')
            fig2 = plt.gcf()
            st.pyplot(fig2)
            plt.clf()

        model_name = f"{model_type.capitalize()} model on {dataset_name} at {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        database.save_model_metadata(
            name=model_name,
            dataset_name=dataset_name,
            model_type=model_type,
            metric_name=metric_name or 'Unknown Metric',
            metric_value=metric_value or 0.0,
            description=f"Best trained model saved on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            created_at=datetime.now(),
        )

        st.success(f"Training completed. Model saved as '{model_name}'.")
        st.write("## Best Model Summary")
        st.write(model)

        if metric_name and metric_value:
            st.metric(label=metric_name, value=f"{metric_value:.4f}")

        if additional_metrics:
            metric_cols = st.columns(len(additional_metrics))
            for idx, (metric, val) in enumerate(additional_metrics.items()):
                metric_cols[idx].metric(label=metric, value=f"{val:.4f}")

    except Exception as e:
        st.warning(f"Could not extract/save performance metric or render charts: {e}")

# Sidebar - Upload Section
st.sidebar.header("Upload or Load Dataset")

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])
load_from_db = st.sidebar.checkbox("Load most recent dataset from database")

if uploaded_file:
    df = data_processing.load_data(uploaded_file)
    if df is not None:
        st.session_state['data'] = df
        default_table_name = uploaded_file.name.replace('.', '_').lower()
        st.session_state['table_name'] = save_uploaded_dataset(df, default_table_name)
elif load_from_db:
    # Load most recent dataset metadata
    meta = database.get_all_dataset_metadata()
    if meta is not None and not meta.empty:
        latest_dataset_name = meta.iloc[0]['name']
        df = database.load_dataframe(latest_dataset_name)
        if df is not None:
            st.session_state['data'] = df
            st.session_state['table_name'] = latest_dataset_name
            st.sidebar.success(f"Loaded dataset '{latest_dataset_name}' from database")
        else:
            st.sidebar.error("Failed to load dataset from database")
    else:
        st.sidebar.info("No datasets found in database")

data = st.session_state.get('data', None)
if data is not None:
    st.write(f"### Loaded Dataset: {st.session_state.get('table_name', 'N/A')} ({data.shape[0]} rows, {data.shape[1]} columns)")

    # Variable selection
    st.header("Select Target and Features")
    columns = data.columns.tolist()
    target = st.selectbox("Select target column (leave blank if clustering)", options=[""] + columns)
    target = target if target != "" else None
    st.session_state['target'] = target

    feature_candidates = [col for col in columns if col != target]
    selected_features = st.multiselect("Select feature columns", feature_candidates, default=feature_candidates)
    st.session_state['selected_features'] = selected_features

    # Model type selection
    model_type = st.radio("Select Model Type", options=['classification', 'regression', 'clustering'])
    st.session_state['model_type'] = model_type

    # Train button
    if st.button("Train & Compare Models"):
        if model_type != 'clustering' and (target is None or target == ""):
            st.error("Please select a target column for classification or regression.")
        elif len(selected_features) == 0:
            st.error("Please select at least one feature column.")
        else:
            dataset_name = st.session_state.get('table_name', f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            train_and_save_best_model(data, target, model_type, selected_features, dataset_name)

    # Show EDA option
    if st.checkbox("Explore Dataset (EDA)"):
        eda.perform_eda(data)

    # Prediction input form & display
    trained_model = st.session_state.get('trained_model', None)
    if trained_model is not None:
        st.header("Predict on New Data")
    
        # Use a unique key for the form
        input_df = predictions.input_new_data_dynamic(data, selected_features, form_key="new_data_form_1")
        if not input_df.empty:
            preds = predictions.predict_with_model(trained_model, input_df, model_type)
            if preds:
                st.subheader("Prediction Results")
                if model_type == 'classification':
                    predicted_class = preds.get('predicted_class')
                    confidence_score = preds.get('probability')
    
                    st.write(f"**Predicted Class:** {predicted_class}")
                    if confidence_score is not None:
                        try:
                            confidence_value = float(confidence_score)
                            st.write(f"**Confidence:** {confidence_value * 100:.2f}%")
                        except (ValueError, TypeError):
                            st.write(f"**Confidence:** {confidence_score}")
    
                    # Show accuracy saved from training
                    accuracy = st.session_state.get('accuracy', None)
                    if accuracy is not None:
                        st.write(f"**Model Accuracy:** {accuracy:.2f}")
    
                elif model_type == 'regression':
                    predicted_value = preds.get('predicted_value')
                    if predicted_value is not None:
                        st.write(f"**Predicted Value:** {predicted_value:.4f}")
    
                    # Optionally, show additional metrics if you have them
                    r2_score = st.session_state.get('r2_score', None)
                    if r2_score is not None:
                        st.write(f"**R2 Score:** {r2_score:.4f}")
    
                elif model_type == 'clustering':
                    predicted_cluster = preds.get('predicted_cluster')
                    if predicted_cluster is not None:
                        st.write(f"**Predicted Cluster:** {predicted_cluster}")
    
                else:
                    st.write(preds)
    
else:
    st.info("Please upload a dataset or load one from the database to get started.")
