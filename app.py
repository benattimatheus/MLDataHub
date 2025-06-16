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

st.set_page_config(page_title="ML Data HUB", layout="wide")
st.title("Aprendizado de Máquina Dinâmico")

# Inicializa as tabelas do banco de dados
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
    st.sidebar.write("### Salvar Conjunto de Dados no Banco")
    table_name_input = st.sidebar.text_input("Digite o nome da tabela do conjunto de dados", value=default_name)
    if st.sidebar.button("Salvar Conjunto de Dados"):
        if not table_name_input.strip():
            st.sidebar.error("Por favor, digite um nome válido para a tabela antes de salvar.")
            return ""
        if database.save_dataframe(df, table_name_input):
            database.save_dataset_metadata(
                name=table_name_input,
                ds_type=st.session_state.get('model_type') or "desconhecido",
                num_rows=df.shape[0],
                num_cols=df.shape[1],
                description=f"Conjunto de dados salvo em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                created_at=datetime.now()
            )
            st.sidebar.success(f"Conjunto de dados salvo com sucesso como '{table_name_input}'.")
            return table_name_input
        else:
            st.sidebar.error(f"Falha ao salvar o conjunto de dados como '{table_name_input}'.")
    return ""

def train_and_save_best_model(df: pd.DataFrame, target: str, model_type: str,
                              selected_features: list, dataset_name: str):
    st.write("### Treinando e Comparando Modelos... Por favor, aguarde.")
    cols = selected_features + ([target] if target else [])
    training_data = df[cols]

    if training_data.isnull().values.any():
        st.error("O conjunto de dados contém valores ausentes. Por favor, limpe os dados antes de treinar.")
        return

    model = None
    exp = None
    results_df = None

    try:
        if model_type == 'classificação':
            model, exp = train_classification_model(training_data, target)
            results_df = pull_classification()
        elif model_type == 'regressão':
            regression_result = train_regression_models(training_data, target)
            if regression_result:
                model = regression_result['pipeline']
                results_df = regression_result['result_df']
                st.session_state['regression_result'] = regression_result
                exp = None
        elif model_type == 'clusterização':
            st.write("### Determinando o Número Ótimo de Clusters")
            determine_optimal_clusters(training_data)

            selected_algorithm = st.session_state.get('selected_algorithm', 'kmeans')
            num_clusters = st.session_state.get('num_clusters', 3)

            model, exp, clustered_data = train_clustering_model(
                training_data,
                model_name=selected_algorithm,
                num_clusters=num_clusters
            )

            results_df = None
        else:
            st.error(f"Tipo de modelo desconhecido: {model_type}")
            return

        if model is None:
            st.error("Falha ao treinar o modelo.")
            return

        st.session_state['trained_model'] = model
        st.session_state['pycaret_exp'] = exp

        if results_df is not None and not results_df.empty:
            with st.expander("Visualizar Resultados Completos da Comparação de Modelos"):
                st.dataframe(results_df.style.format(precision=4))

        additional_metrics = {}
        metric_name, metric_value = None, None

        if model_type == 'classificação':
            metric_name = 'Acurácia'
            metric_value = results_df.at[results_df.index[0], 'Accuracy'] if 'Accuracy' in results_df else None
            for m in ['AUC', 'Recall', 'Precision', 'F1']:
                if m in results_df:
                    additional_metrics[m] = results_df.at[results_df.index[0], m]

            st.write("### Gráficos de Desempenho em Classificação")
            plot_model_classification(model, plot='confusion_matrix', display_format='streamlit')
            plot_model_classification(model, plot='auc', display_format='streamlit')
            
        elif model_type == 'regressão':
            st.write("### Melhorando o Modelo de Regressão...")
            regression_result = st.session_state.get('regression_result')
        
            if regression_result:
                y = regression_result['y']
                y_pred = regression_result['y_pred']
                model_name = regression_result['model_name']
        
                metric_name = 'R²'
                metric_value = ((pd.Series(y_pred) - pd.Series(y))**2).mean()
                st.session_state['r2_score'] = metric_value
        
                st.write("### Gráficos de Desempenho em Regressão")
        
                st.write("#### Valores Reais vs. Preditos")
                chart_data = pd.DataFrame({'Real': y, 'Previsto': y_pred})
                st.line_chart(chart_data)
        
                st.write("#### Gráfico de Resíduos")
                residuals = y - y_pred
                residuals_data = pd.DataFrame({'Previsto': y_pred, 'Resíduos': residuals})
                st.line_chart(residuals_data.set_index('Previsto'))
        
                st.write("#### Histograma de Resíduos")
                st.bar_chart(pd.Series(residuals).value_counts())
        
        elif model_type == 'clusterização':
            st.write("### Gráficos de Desempenho em Clusterização")
            for plot in ['silhouette', 'cluster']:
                try:
                    plot_model_clustering(model, plot=plot, display_format='streamlit')
                except Exception as e:
                    st.warning(f"Não foi possível gerar o gráfico {plot}: {e}")

        if model_type == 'clusterização':
            model_name = f"{model_type.capitalize()} ({selected_algorithm}) em {dataset_name} em {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            model_name = f"Modelo de {model_type.capitalize()} em {dataset_name} em {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        database.save_model_metadata(
            name=model_name,
            dataset_name=dataset_name,
            model_type=model_type,
            metric_name=metric_name or 'Desconhecido',
            metric_value=metric_value or 0.0,
            description=f"Modelo salvo em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            created_at=datetime.now()
        )

        st.success(f"Treinamento concluído. Modelo salvo como '{model_name}'.")
        st.write("## Resumo do Melhor Modelo")
        st.write(model)

        if metric_name and metric_value:
            st.metric(label=metric_name, value=f"{metric_value:.4f}")
        if additional_metrics:
            cols = st.columns(len(additional_metrics))
            for i, (k, v) in enumerate(additional_metrics.items()):
                cols[i].metric(label=k, value=f"{v:.4f}")

    except Exception as e:
        st.warning(f"Erro durante o treinamento ou visualização: {e}")

# Barra lateral
st.sidebar.header("Carregar ou Selecionar Conjunto de Dados")
uploaded_file = st.sidebar.file_uploader("Carregar conjunto de dados (CSV ou Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = data_processing.load_data(uploaded_file)
    default_table_name = uploaded_file.name.replace('.', '_').lower()
    df = data_processing.clean_data(df)  # Limpar antes de salvar
    st.session_state['data'] = df
    st.session_state['table_name'] = save_uploaded_dataset(df, default_table_name)

data = st.session_state['data']
if data is not None:
    st.write(f"### Conjunto de Dados Carregado: {st.session_state['table_name']} ({data.shape[0]} linhas, {data.shape[1]} colunas)")

    st.header("Selecione o Alvo e as Variáveis")
    columns = data.columns.tolist()
    target = st.selectbox("Selecione a coluna alvo (deixe em branco para clusterização)", [""] + columns, key="target_select")
    target = target if target else None
    st.session_state['target'] = target

    features = [col for col in columns if col != target]
    selected_features = st.multiselect("Selecione as colunas de variáveis", features, default=features)
    st.session_state['selected_features'] = selected_features

    model_type = st.radio("Selecione o Tipo de Modelo", ['classificação', 'regressão', 'clusterização'], key="model_radio")
    st.session_state['model_type'] = model_type

    if model_type == 'clusterização':
        clustering_algorithms = ['kmeans', 'birch']
        selected_algorithm = st.selectbox("Selecione o Algoritmo de Clusterização", clustering_algorithms, index=clustering_algorithms.index('kmeans'))
        num_clusters = st.slider("Número de Clusters (apenas para kmeans)", 2, 15, 3)
        st.session_state['selected_algorithm'] = selected_algorithm
        st.session_state['num_clusters'] = num_clusters

    if st.button("Treinar e Comparar Modelos"):
        if model_type != 'clusterização' and not target:
            st.error("Por favor, selecione uma coluna alvo.")
        elif not selected_features:
            st.error("Por favor, selecione pelo menos uma coluna de variável.")
        else:
            name = st.session_state['table_name'] or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            train_and_save_best_model(data, target, model_type, selected_features, name)

    if st.checkbox("Explorar o Conjunto de Dados (EDA)"):
        eda.perform_eda(data)

    model = st.session_state['trained_model']
    if model is not None:
        st.header("Prever com Novos Dados")
        input_df = predictions.input_new_data_dynamic(data, selected_features, form_key="new_data_form")
        if not input_df.empty:
            preds = predictions.predict_with_model(model, input_df, model_type)  # Chama função de previsão atualizada
            predicted_class = preds['predicted_class']  # Classe prevista
            confidence = preds['probability']  # Probabilidade
            st.write(f"**Classe Prevista:** {predicted_class}")
            st.write(f"**Confiança:** {confidence * 100:.2f}%")
            if preds:
                st.subheader("Resultados da Previsão")
                if model_type == 'classificação':
                    predicted_class = preds['predicted_class']  # Classe prevista
                    confidence = preds['probability']  # Probabilidade
                    st.write(f"**Classe Prevista:** {predicted_class}")
                    st.write(f"**Confiança:** {confidence * 100:.2f}%")
                elif model_type == 'regressão':
                    st.write(f"**Valor Previsto:** {preds.get('predicted_value'):.4f}")
                    r2 = st.session_state.get('r2_score')
                    if r2:
                        st.write(f"**R²:** {r2:.4f}")
                elif model_type == 'clusterização':
                    st.write(f"**Cluster Previsto:** {preds.get('predicted_cluster')}")

else:
    st.info("Por favor, carregue um conjunto de dados ou selecione um do banco para iniciar.")
