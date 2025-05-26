import sqlite3
import pandas as pd

DB_PATH = "data/predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS predictions (model_type TEXT, input TEXT, prediction TEXT)")
    conn.close()

def save_prediction(model_type, input_data, prediction):
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame({'model_type': [model_type], 'input': [str(input_data)], 'prediction': [str(prediction)]})
    df.to_sql('predictions', conn, if_exists='append', index=False)
    conn.close()
