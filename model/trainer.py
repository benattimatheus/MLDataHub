from pycaret.classification import setup as cls_setup, compare_models as cls_compare, finalize_model as cls_finalize, save_model as cls_save
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, finalize_model as reg_finalize, save_model as reg_save
from pycaret.clustering import setup as clu_setup, create_model as clu_create, assign_model, save_model as clu_save

def train_classification(df):
    cls_setup(data=df, target='target', session_id=42, verbose=False)
    best = cls_compare()
    final = cls_finalize(best)
    cls_save(final, 'data/classification_model')
    return final

def train_regression(df):
    reg_setup(data=df, target='target', session_id=42, verbose=False)
    best = reg_compare()
    final = reg_finalize(best)
    reg_save(final, 'data/regression_model')
    return final

def train_clustering(df):
    clu_setup(data=df, session_id=42, verbose=False)
    model = clu_create('kmeans')
    df_clustered = assign_model(model)
    clu_save(model, 'data/clustering_model')
    return df_clustered
