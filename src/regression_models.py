import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint, uniform
from src.data_processing import remove_outliers_iqr
import streamlit as st

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

def detect_text_columns(df: pd.DataFrame, exclude_cols: list = []):
    text_cols = []
    for col in df.select_dtypes(include='object').columns:
        if col in exclude_cols:
            continue
        avg_len = df[col].dropna().astype(str).map(len).mean()
        if avg_len > 20:
            text_cols.append(col)
    return text_cols

def train_regression_models(
    df: pd.DataFrame,
    target: str,
    cat_features: list = None,
    text_features: list = None,
    numeric_features: list = None,
    max_tfidf_features: int = 1000,
    custom_feature_engineering: callable = None
):
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    class TextConcatenator(BaseEstimator, TransformerMixin):
        def __init__(self, text_columns):
            self.text_columns = text_columns

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.text_columns].fillna('').apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    df = df.copy()
    df = remove_outliers_iqr(df, target)

    if numeric_features is None:
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_features:
            numeric_features.remove(target)

    if cat_features is None:
        cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if text_features is None:
            text_features = detect_text_columns(df, exclude_cols=cat_features)
        for txt_col in text_features:
            if txt_col in cat_features:
                cat_features.remove(txt_col)

    if text_features is None:
        text_features = detect_text_columns(df, exclude_cols=cat_features)

    st.write(f"Numéricas usadas: {numeric_features}")
    st.write(f"Categóricas usadas: {cat_features}")
    st.write(f"Textuais usadas: {text_features}")

    if custom_feature_engineering is not None:
        df = custom_feature_engineering(df)

    # Check if all required columns are present
    required_columns = numeric_features + cat_features + text_features + [target]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        return None, None

    X = df.drop(columns=[target])
    y = df[target]

    num_transformer = StandardScaler()

    transformers = []
    if len(numeric_features) > 0:
        transformers.append(('num', num_transformer, numeric_features))
    if len(cat_features) > 0:
        cat_pipe = make_pipeline(TargetEncoder(cols=cat_features))
        transformers.append(('cat_target', cat_pipe, cat_features))
    if len(text_features) > 0:
        text_pipe = Pipeline([
            ('concat', TextConcatenator(text_features)),
            ('tfidf', TfidfVectorizer(max_features=max_tfidf_features, ngram_range=(1,3), stop_words='english'))
        ])
        transformers.append(('text', text_pipe, text_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    modelos = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        'ElasticNet': ElasticNet(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42),
    }
    if CATBOOST_AVAILABLE:
        modelos['CatBoost'] = CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)

    stacked_model = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(random_state=42)),
            ('gbr', GradientBoostingRegressor(random_state=42)),
            ('xgb', xgb.XGBRegressor(random_state=42, verbosity=0))
        ],
        final_estimator=LinearRegression()
    )
    modelos['Stacked'] = stacked_model

    param_distributions = {
        'Random Forest': {
            'regressor__n_estimators': randint(100, 500),
            'regressor__max_depth': randint(3, 20),
            'regressor__min_samples_split': randint(2, 20),
            'regressor__min_samples_leaf': randint(1, 10),
            'regressor__max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7, 1.0],
        },
        'Gradient Boosting': {
            'regressor__n_estimators': randint(100, 500),
            'regressor__learning_rate': uniform(0.01, 0.3),
            'regressor__max_depth': randint(3, 10),
            'regressor__min_samples_split': randint(2, 20),
            'regressor__min_samples_leaf': randint(1, 10),
            'regressor__subsample': uniform(0.6, 0.4),
            'regressor__max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7, 1.0],
        },
        'XGBoost': {
            'regressor__n_estimators': randint(100, 500),
            'regressor__learning_rate': uniform(0.01, 0.3),
            'regressor__max_depth': randint(3, 15),
            'regressor__subsample': uniform(0.6, 0.4),
            'regressor__colsample_bytree': uniform(0.6, 0.4),
            'regressor__gamma': uniform(0, 5),
            'regressor__reg_alpha': uniform(0, 1),
            'regressor__reg_lambda': uniform(0, 1),
        },
        'CatBoost': {
            'regressor__iterations': randint(100, 500),
            'regressor__depth': randint(3, 10),
            'regressor__learning_rate': uniform(0.01, 0.3),
            'regressor__l2_leaf_reg': uniform(1, 10),
            'regressor__border_count': randint(32, 255),
        }
    }

    resultados = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for nome, modelo in modelos.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', modelo)
        ])

        st.write(f"Treinando e avaliando {nome}...")

        if nome in param_distributions:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions[nome],
                n_iter=50,
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            search.fit(X, y)
            best_pipeline = search.best_estimator_
            r2 = search.best_score_
        else:
            scores = cross_val_score(pipeline, X, y, scoring='r2', cv=cv, n_jobs=-1)
            pipeline.fit(X, y)
            r2 = scores.mean()
            best_pipeline = pipeline

        y_pred = best_pipeline.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        resultados.append({
            'Modelo': nome,
            'MAE': mae,
            'RMSE': rmse,
            'R2 (CV média)': r2,
            'pipeline': best_pipeline
        })

    resultado_df = pd.DataFrame([{
        'Modelo': r['Modelo'],
        'MAE': r['MAE'],
        'RMSE': r['RMSE'],
        'R2 (CV média)': r['R2 (CV média)']
    } for r in resultados]).sort_values(by='R2 (CV média)', ascending=False)

    st.write("## Resultados dos modelos")
    st.dataframe(resultado_df)

    melhor_resultado = max(resultados, key=lambda x: x['R2 (CV média)'])
    melhor_pipeline = melhor_resultado['pipeline']

    st.success(f"Melhor modelo: {melhor_resultado['Modelo']} com R² médio de {melhor_resultado['R2 (CV média)']:.4f}")

    return melhor_pipeline, resultado_df
