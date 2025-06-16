import streamlit as st
import pandas as pd
from datetime import datetime
from src import data_processing, eda, predictions, database
import matplotlib.pyplot as plt

from pycaret.classification import plot_model as plot_model_classification, pull as pull_classification
from pycaret.regression import plot_model as plot_model_regression, pull as pull_regression
from pycaret.clustering import plot_model as plot_model_clustering

from src.classification_models import train_classification_model
from src.regression_models import train_regression_models
from src.clustering_models import train_clustering_model, determine_optimal_clusters

# Estilo personalizado
st.set_page_config(page_title="🔍 ML Data HUB", layout="wide", page_icon="📊")

st.markdown("""
    <style>
        .main {background-color: #f4f6f9;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>input, .stSelectbox>div>div {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 ML Data HUB")
st.subheader("Análise de Dados e Treinamento de Modelos com Interface Dinâmica")

# Inicializa tabelas
database.initialize_tables()

# Estado padrão da sessão
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
    st.sidebar.write("📁 **Salvar Conjunto de Dados no Banco**")
    table_name_input = st.sidebar.text_input("Digite o nome da tabela", value=default_name)
    if st.sidebar.button("💾 Salvar"):
        if not table_name_input.strip():
            st.sidebar.error("Digite um nome válido para a tabela.")
            return ""
        if database.save_dataframe(df, table_name_input):
            database.save_dataset_metadata(
                name=table_name_input,
                ds_type=st.session_state.get('model_type') or "desconhecido",
                num_rows=df.shape[0],
                num_cols=df.shape[1],
                description=f"Conjunto salvo em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                created_at=datetime.now()
            )
            st.sidebar.success(f"Conjunto salvo como '{table_name_input}'.")
            return table_name_input
        else:
            st.sidebar.error(f"Erro ao salvar '{table_name_input}'.")
    return ""

def train_and_save_best_model(df, target, model_type, selected_features, dataset_name):
    st.info("⚙️ Treinando e comparando modelos...")
    cols = selected_features + ([target] if target else [])
    training_data = df[cols]

    if training_data.isnull().values.any():
        st.error("❗ Dados com valores ausentes. Corrija antes de treinar.")
        return

    model = exp = results_df = None

    try:
        if model_type == 'classificação':
            model, exp = train_classification_model(training_data, target)
            results_df = pull_classification()

        elif model_type == 'regressão':
            result = train_regression_models(training_data, target)
            if result:
                model = result['pipeline']
                results_df = result['result_df']
                st.session_state['regression_result'] = result

        elif model_type == 'clusterização':
            st.write("🔎 Determinando o número ótimo de clusters...")
            determine_optimal_clusters(training_data)
            selected_algorithm = st.session_state.get('selected_algorithm', 'kmeans')
            num_clusters = st.session_state.get('num_clusters', 3)
            model, exp, _ = train_clustering_model(training_data, selected_algorithm, num_clusters)

        if model is None:
            st.error("❗ Falha ao treinar o modelo.")
            return

        st.session_state['trained_model'] = model
        st.session_state['pycaret_exp'] = exp

        if results_df is not None and not results_df.empty:
            with st.expander("📈 Visualizar Resultados da Comparação"):
                st.dataframe(results_df.style.format(precision=4))

        metric_name = metric_value = None
        additional_metrics = {}

        if model_type == 'classificação':
            metric_name = 'Acurácia'
            metric_value = results_df.at[results_df.index[0], 'Accuracy'] if 'Accuracy' in results_df else None
            for m in ['AUC', 'Recall', 'Precision', 'F1']:
                if m in results_df:
                    additional_metrics[m] = results_df.at[results_df.index[0], m]

            st.write("📊 Gráficos de Desempenho:")
            plot_model_classification(model, plot='confusion_matrix', display_format='streamlit')
            plot_model_classification(model, plot='auc', display_format='streamlit')

        elif model_type == 'regressão':
            result = st.session_state['regression_result']
            y, y_pred = result['y'], result['y_pred']
            residuals = y - y_pred

            metric_name = 'R²'
            metric_value = ((pd.Series(y_pred) - pd.Series(y))**2).mean()
            st.session_state['r2_score'] = metric_value

            st.write("📉 Gráficos de Regressão:")
            st.line_chart(pd.DataFrame({'Real': y, 'Previsto': y_pred}))
            st.line_chart(pd.DataFrame({'Resíduos': residuals}).set_index(pd.Series(y_pred)))
            st.bar_chart(pd.Series(residuals).value_counts())

        elif model_type == 'clusterização':
            for plot in ['silhouette', 'cluster']:
                try:
                    plot_model_clustering(model, plot=plot, display_format='streamlit')
                except Exception as e:
                    st.warning(f"Erro no gráfico '{plot}': {e}")

        model_name = f"{model_type.capitalize()} - {dataset_name} ({datetime.now().strftime('%H%M%S')})"
        database.save_model_metadata(
            name=model_name,
            dataset_name=dataset_name,
            model_type=model_type,
            metric_name=metric_name or "N/A",
            metric_value=metric_value or 0.0,
            description=f"Treinado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            created_at=datetime.now()
        )

        st.success(f"✅ Modelo salvo como '{model_name}'.")
        st.write("📌 **Resumo do Melhor Modelo**")
        st.write(model)

        if metric_name and metric_value:
            st.metric(label=metric_name, value=f"{metric_value:.4f}")
        if additional_metrics:
            cols = st.columns(len(additional_metrics))
            for i, (k, v) in enumerate(additional_metrics.items()):
                cols[i].metric(label=k, value=f"{v:.4f}")

    except Exception as e:
        st.error(f"❗ Erro ao treinar: {e}")

# Sidebar
st.sidebar.header("📂 Dados")
uploaded_file = st.sidebar.file_uploader("Carregar dataset (CSV/Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = data_processing.load_data(uploaded_file)
    df = data_processing.clean_data(df)
    st.session_state['data'] = df
    st.session_state['table_name'] = save_uploaded_dataset(df, uploaded_file.name.replace('.', '_').lower())

data = st.session_state['data']
if data is not None:
    st.markdown("----")
    st.write(f"✅ **Dados Carregados:** `{st.session_state['table_name']}` ({data.shape[0]} linhas, {data.shape[1]} colunas)")

    st.header("🧪 Seleção de Variáveis")
    columns = data.columns.tolist()
    target = st.selectbox("🎯 Coluna alvo (vazia para clusterização)", [""] + columns)
    target = target or None
    st.session_state['target'] = target

    features = [c for c in columns if c != target]
    selected_features = st.multiselect("📌 Variáveis preditoras", features, default=features)
    st.session_state['selected_features'] = selected_features

    model_type = st.radio("🧠 Tipo de modelo", ['classificação', 'regressão', 'clusterização'])
    st.session_state['model_type'] = model_type

    if model_type == 'clusterização':
        st.subheader("⚙️ Parâmetros de Clusterização")
        st.session_state['selected_algorithm'] = st.selectbox("Algoritmo", ['kmeans', 'birch'])
        st.session_state['num_clusters'] = st.slider("Número de clusters (k-means)", 2, 15, 3)

    if st.button("🚀 Treinar e Comparar"):
        if model_type != 'clusterização' and not target:
            st.error("🎯 Selecione a variável alvo.")
        elif not selected_features:
            st.error("📌 Selecione as variáveis preditoras.")
        else:
            name = st.session_state['table_name'] or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            train_and_save_best_model(data, target, model_type, selected_features, name)

    if st.checkbox("🔍 Análise Exploratória (EDA)"):
        eda.perform_eda(data)

    if st.session_state['trained_model'] is not None:
        st.header("📤 Fazer Previsões")
        input_df = predictions.input_new_data_dynamic(data, selected_features, form_key="new_data_form")
        if not input_df.empty:
            preds = predictions.predict_with_model(st.session_state['trained_model'], input_df, model_type)
            if preds:
                st.subheader("📢 Resultado da Previsão")
                if model_type == 'classificação':
                    st.success(f"🧾 Classe Prevista: `{preds['predicted_class']}` com {preds['probability'] * 100:.2f}% de confiança.")
                elif model_type == 'regressão':
                    st.success(f"📉 Valor Previsto: `{preds['predicted_value']:.4f}`")
                    r2 = st.session_state.get('r2_score')
                    if r2:
                        st.info(f"R²: `{r2:.4f}`")
                elif model_type == 'clusterização':
                    st.success(f"🔗 Cluster Previsto: `{preds['predicted_cluster']}`")
else:
    st.info("📥 Carregue um conjunto de dados para começar.")
