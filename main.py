import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import TargetEncoder

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    print("\nColunas disponíveis:", list(df.columns))
    print("\nResumo estatístico:")
    print(df.describe(include='all'))
    return df

def preparar_dados(df, target_column):
    df = df.copy()
    ano_atual = pd.Timestamp.now().year
    df['car_age'] = ano_atual - df['year']

    # Log-transform em km_driven para reduzir skew
    df['km_driven_log'] = np.log1p(df['km_driven'])

    # Feature binária se é First Owner
    df['is_first_owner'] = (df['owner'] == 'First Owner').astype(int)

    # Binário para transmissão automática
    df['is_automatic'] = (df['transmission'] == 'Automatic').astype(int)

    # Remover outliers do target usando IQR
    y = df[target_column]
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    mask = (y >= limite_inferior) & (y <= limite_superior)
    df = df[mask]

    # Atualizar X e y após remoção outliers
    X = df.drop(columns=[target_column, 'year', 'km_driven', 'owner', 'transmission'])
    y = df[target_column]

    # Criar features de interação simples
    X['age_km_interaction'] = X['car_age'] * X['km_driven_log']
    X['owner_auto_interaction'] = X['is_first_owner'] * X['is_automatic']

    return X, y

def avaliar_modelos(X, y):
    cat_features = ['fuel', 'seller_type']
    num_features = ['car_age', 'km_driven_log', 'is_first_owner', 'is_automatic', 'age_km_interaction', 'owner_auto_interaction']
    text_feature = 'name'

    # Pré-processamento numérico
    num_transformer = StandardScaler()

    # Encoding categórico (target encoding)
    target_encoder = TargetEncoder()

    # Pipeline para texto com TF-IDF usando bigrams e max_features 500 para mais riqueza
    tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    text_transformer = Pipeline([
        ('tfidf', tfidf_vectorizer)
    ])

    # Preprocessor combinando tudo
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', target_encoder, cat_features),
        ('text', text_transformer, text_feature)
    ], remainder='drop')

    modelos = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
    }

    if CATBOOST_AVAILABLE:
        modelos['CatBoost'] = CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)

    resultados = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for nome, modelo in modelos.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', modelo)
        ])

        print(f"\nTreinando e avaliando {nome}...")
        scores = cross_val_score(pipeline, X, y, scoring='r2', cv=cv, n_jobs=-1)

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = scores.mean()

        resultados.append({
            'Modelo': nome,
            'MAE': mae,
            'RMSE': rmse,
            'R² (CV média)': r2
        })

    resultado_df = pd.DataFrame(resultados).sort_values(by='R² (CV média)', ascending=False)
    print("\nResultados dos Modelos (com validação cruzada, TF-IDF bigrams e interações):")
    print(resultado_df)

    sns.barplot(x='R² (CV média)', y='Modelo', data=resultado_df)
    plt.title('Comparação de Modelos - R² (média CV)')
    plt.show()

    melhor_modelo = resultado_df.iloc[0]
    print(f"\n🏆 Melhor modelo: {melhor_modelo['Modelo']} com R² médio de {melhor_modelo['R² (CV média)']:.4f}")

if __name__ == "__main__":
    df = carregar_dados("data/regression/CAR DETAILS FROM CAR DEKHO.csv")
    target_column = "selling_price"
    X, y = preparar_dados(df, target_column)
    avaliar_modelos(X, y)
