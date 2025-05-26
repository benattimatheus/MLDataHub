from sklearn.datasets import load_diabetes, load_breast_cancer, load_wine
import pandas as pd

def load_regression_data():
    data = load_diabetes(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df

def load_classification_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df['target'] = data.target
    return df

def load_clustering_data():
    data = load_wine(as_frame=True)
    df = data.frame.drop(columns=['target'])
    return df
