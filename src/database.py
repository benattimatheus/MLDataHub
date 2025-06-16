import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Float
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import sessionmaker
from typing import Optional
from datetime import datetime
import logging
import streamlit as st  # Para relatórios de erro no contexto do Streamlit, se necessário

# Configurar logging para depuração e erros
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL de conexão padrão do banco de dados (pode ser alterada para qualquer URI suportada pelo SQLAlchemy)
DATABASE_URL = "sqlite:///ml_app.db"

_engine = None
_Session = None

def init_engine(database_url: str = DATABASE_URL):
    """
    Inicializa e retorna o engine do SQLAlchemy e o sessionmaker.
    Apenas cria um engine/sessão por ciclo de vida do aplicativo.
    """
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(database_url, echo=False, future=True)
        _Session = sessionmaker(bind=_engine)
        logger.info(f"Engine do banco de dados criado com a URL: {database_url}")
    return _engine, _Session

def save_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
    """
    Salva um DataFrame do Pandas em uma tabela SQL.

    Args:
        df: DataFrame a ser salvo.
        table_name: Nome da tabela SQL.
        if_exists: Ação se a tabela existir: 'fail', 'replace' ou 'append'.

    Returns:
        True se salvo com sucesso, False caso contrário.
    """
    engine, _ = init_engine()
    try:
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        logger.info(f"DataFrame salvo em '{table_name}' com forma {df.shape}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Falha ao salvar DataFrame em '{table_name}': {e}")
        if st:
            st.error(f"Erro ao salvar no banco de dados: {e}")
        return False

def load_dataframe(table_name: str) -> Optional[pd.DataFrame]:
    """
    Carrega uma tabela SQL em um DataFrame do Pandas.

    Args:
        table_name: Nome da tabela SQL a ser carregada.

    Returns:
        DataFrame em caso de sucesso, None em caso de falha.
    """
    engine, _ = init_engine()
    try:
        df = pd.read_sql_table(table_name, con=engine)
        logger.info(f"Tabela '{table_name}' carregada com forma {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Falha ao carregar a tabela '{table_name}': {e}")
        if st:
            st.error(f"Erro ao carregar tabela do banco de dados: {e}")
        return None

def initialize_tables():
    """
    Cria tabelas de metadados necessárias para conjuntos de dados e modelos, se não existirem.
    """
    engine, _ = init_engine()
    metadata = MetaData()

    datasets_table = Table(
        'datasets', metadata,
        Column('id', Integer, primary_key=True, autoincrement=True),
        Column('name', String(255), nullable=False, unique=True),
        Column('description', Text, nullable=True),
        Column('type', String(50), nullable=False),  # e.g., regressão/classificação/clusterização
        Column('num_rows', Integer, nullable=True),
        Column('num_cols', Integer, nullable=True),
        Column('created_at', DateTime, nullable=False),
    )

    models_table = Table(
        'models', metadata,
        Column('id', Integer, primary_key=True, autoincrement=True),
        Column('name', String(255), nullable=False, unique=True),
        Column('dataset_name', String(255), nullable=False),
        Column('description', Text, nullable=True),
        Column('model_type', String(50), nullable=False),
        Column('metric_name', String(100), nullable=True),
        Column('metric_value', Float, nullable=True),
        Column('created_at', DateTime, nullable=False),
    )

    try:
        metadata.create_all(engine)
        logger.info("Tabelas do banco de dados inicializadas para conjuntos de dados e modelos")
    except SQLAlchemyError as e:
        logger.error(f"Erro ao inicializar tabelas: {e}")
        if st:
            st.error(f"Erro na inicialização do banco de dados: {e}")

def save_dataset_metadata(
    name: str,
    ds_type: str,
    num_rows: int,
    num_cols: int,
    description: str = "",
    created_at: Optional[datetime] = None
) -> bool:
    """
    Salva informações de metadados sobre um conjunto de dados.

    Args:
        name: Identificador do nome do conjunto de dados.
        ds_type: Tipo de conjunto de dados, e.g., regressão, classificação, clusterização.
        num_rows: Número de linhas.
        num_cols: Número de colunas.
        description: Texto de descrição opcional.
        created_at: Timestamp, padrão é agora.

    Returns:
        True se salvo com sucesso, caso contrário False.
    """
    engine, Session = init_engine()
    session = Session()
    if created_at is None:
        created_at = datetime.now()

    metadata = MetaData()
    datasets_table = Table('datasets', metadata, autoload_with=engine)

    try:
        ins = datasets_table.insert().values(
            name=name,
            description=description,
            type=ds_type,
            num_rows=num_rows,
            num_cols=num_cols,
            created_at=created_at,
        )
        session.execute(ins)
        session.commit()
        logger.info(f"Metadados do conjunto de dados '{name}' salvos")
        return True
    except IntegrityError:
        session.rollback()
        logger.warning(f"Metadados do conjunto de dados '{name}' já existem")
        return False
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Erro ao salvar metadados do conjunto de dados: {e}")
        if st:
            st.error(f"Erro ao salvar metadados do conjunto de dados: {e}")
        return False
    finally:
        session.close()

def save_model_metadata(
    name: str,
    dataset_name: str,
    model_type: str,
    metric_name: str,
    metric_value: float,
    description: str = "",
    created_at: Optional[datetime] = None
) -> bool:
    """
    Salva informações de metadados sobre um modelo treinado.

    Args:
        name: Nome/identificador do modelo.
        dataset_name: Nome do conjunto de dados associado.
        model_type: Tipo de modelo (regressão/classificação/clusterização).
        metric_name: Nome da métrica de desempenho (e.g., 'Acurácia', 'R²').
        metric_value: Valor da métrica.
        description: Descrição opcional.
        created_at: Timestamp, padrão é agora.

    Returns:
        True se salvo com sucesso, caso contrário False.
    """
    engine, Session = init_engine()
    session = Session()
    if created_at is None:
        created_at = datetime.now()

    metadata = MetaData()
    models_table = Table('models', metadata, autoload_with=engine)

    try:
        ins = models_table.insert().values(
            name=name,
            dataset_name=dataset_name,
            model_type=model_type,
            metric_name=metric_name,
            metric_value=metric_value,
            description=description,
            created_at=created_at,
        )
        session.execute(ins)
        session.commit()
        logger.info(f"Metadados do modelo '{name}' salvos")
        return True
    except IntegrityError:
        session.rollback()
        logger.warning(f"Metadados do modelo '{name}' já existem")
        return False
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Erro ao salvar metadados do modelo: {e}")
        if st:
            st.error(f"Erro ao salvar metadados do modelo: {e}")
        return False
    finally:
        session.close()

def get_all_dataset_metadata() -> Optional[pd.DataFrame]:
    """
    Carrega todos os metadados do conjunto de dados.

    Returns:
        DataFrame dos metadados dos conjuntos de dados ou None em caso de erro.
    """
    engine, _ = init_engine()
    try:
        query = "SELECT * FROM datasets ORDER BY created_at DESC"
        df = pd.read_sql(query, con=engine)
        logger.info(f"Metadados do conjunto de dados carregados, registros: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Falha ao carregar metadados do conjunto de dados: {e}")
        if st:
            st.error(f"Erro ao carregar metadados do conjunto de dados: {e}")
        return None

def get_all_model_metadata() -> Optional[pd.DataFrame]:
    """
    Carrega todos os metadados do modelo.

    Returns:
        DataFrame dos metadados dos modelos ou None em caso de erro.
    """
    engine, _ = init_engine()
    try:
        query = "SELECT * FROM models ORDER BY created_at DESC"
        df = pd.read_sql(query, con=engine)
        logger.info(f"Metadados do modelo carregados, registros: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Falha ao carregar metadados do modelo: {e}")
        if st:
            st.error(f"Erro ao carregar metadados do modelo: {e}")
        return None
