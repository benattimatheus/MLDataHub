import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, Float
)
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import sessionmaker
from typing import Optional
from datetime import datetime
import logging

import streamlit as st  # For error reporting in Streamlit context if needed

# Configure logging for debugging and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default database file connection URL (can be changed to any supported SQLAlchemy URI)
DATABASE_URL = "sqlite:///ml_app.db"

_engine = None
_Session = None

def init_engine(database_url: str = DATABASE_URL):
    """
    Initialize and return the SQLAlchemy engine and sessionmaker.
    Only creates one engine/session per app lifecycle.
    """
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(database_url, echo=False, future=True)
        _Session = sessionmaker(bind=_engine)
        logger.info(f"Database engine created with url: {database_url}")
    return _engine, _Session

def save_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> bool:
    """
    Save a Pandas DataFrame to a SQL table.

    Args:
        df: DataFrame to save.
        table_name: Name of the SQL table.
        if_exists: Action if table exists: 'fail', 'replace', or 'append'.

    Returns:
        True if saved successfully, False otherwise.
    """
    engine, _ = init_engine()
    try:
        df.to_sql(table_name, con=engine, if_exists=if_exists, index=False)
        logger.info(f"Saved DataFrame to '{table_name}' with shape {df.shape}")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Failed to save DataFrame to '{table_name}': {e}")
        if st:
            st.error(f"Database save error: {e}")
        return False

def load_dataframe(table_name: str) -> Optional[pd.DataFrame]:
    """
    Load a SQL table into a Pandas DataFrame.

    Args:
        table_name: Name of the SQL table to load.

    Returns:
        DataFrame on success, None on failure.
    """
    engine, _ = init_engine()
    try:
        df = pd.read_sql_table(table_name, con=engine)
        logger.info(f"Loaded table '{table_name}' with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load table '{table_name}': {e}")
        if st:
            st.error(f"Database load error: {e}")
        return None

def initialize_tables():
    """
    Create required metadata tables for datasets and models if they do not exist.
    """
    engine, _ = init_engine()
    metadata = MetaData()

    datasets_table = Table(
        'datasets', metadata,
        Column('id', Integer, primary_key=True, autoincrement=True),
        Column('name', String(255), nullable=False, unique=True),
        Column('description', Text, nullable=True),
        Column('type', String(50), nullable=False),  # e.g., regression/classification/clustering
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
        logger.info("Initialized database tables for datasets and models")
    except SQLAlchemyError as e:
        logger.error(f"Error initializing tables: {e}")
        if st:
            st.error(f"Database initialization error: {e}")

def save_dataset_metadata(
    name: str,
    ds_type: str,
    num_rows: int,
    num_cols: int,
    description: str = "",
    created_at: Optional[datetime] = None
) -> bool:
    """
    Save metadata info about a dataset.

    Args:
        name: Name identifier for dataset.
        ds_type: Dataset type, e.g., regression, classification, clustering.
        num_rows: Number of rows.
        num_cols: Number of columns.
        description: Optional description text.
        created_at: Timestamp, defaults to now.

    Returns:
        True if saved successfully, else False.
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
        logger.info(f"Dataset metadata '{name}' saved")
        return True
    except IntegrityError:
        session.rollback()
        logger.warning(f"Dataset metadata '{name}' already exists")
        return False
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error saving dataset metadata: {e}")
        if st:
            st.error(f"Dataset metadata save error: {e}")
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
    Save metadata info about a trained model.

    Args:
        name: Model name/identifier.
        dataset_name: Associated dataset name.
        model_type: Type of model (regression/classification/clustering).
        metric_name: Performance metric name (e.g., 'Accuracy', 'R2').
        metric_value: Metric value.
        description: Optional description.
        created_at: Timestamp, defaults to now.

    Returns:
        True if saved successfully, else False.
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
        logger.info(f"Model metadata '{name}' saved")
        return True
    except IntegrityError:
        session.rollback()
        logger.warning(f"Model metadata '{name}' already exists")
        return False
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Error saving model metadata: {e}")
        if st:
            st.error(f"Model metadata save error: {e}")
        return False
    finally:
        session.close()

def get_all_dataset_metadata() -> Optional[pd.DataFrame]:
    """
    Load all dataset metadata.

    Returns:
        DataFrame of datasets metadata or None on error.
    """
    engine, _ = init_engine()
    try:
        query = "SELECT * FROM datasets ORDER BY created_at DESC"
        df = pd.read_sql(query, con=engine)
        logger.info(f"Loaded dataset metadata, records: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset metadata: {e}")
        if st:
            st.error(f"Load dataset metadata error: {e}")
        return None

def get_all_model_metadata() -> Optional[pd.DataFrame]:
    """
    Load all model metadata.

    Returns:
        DataFrame of models metadata or None on error.
    """
    engine, _ = init_engine()
    try:
        query = "SELECT * FROM models ORDER BY created_at DESC"
        df = pd.read_sql(query, con=engine)
        logger.info(f"Loaded model metadata, records: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to load model metadata: {e}")
        if st:
            st.error(f"Load model metadata error: {e}")
        return None
