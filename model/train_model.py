# from pycaret.time_series import setup, compare_models, finalize_model, save_model, pull
# import pandas as pd
# import yfinance as yf

# # Load data
# data = yf.download("AAPL", period="2y")['Close'].reset_index()
# data.columns = ['Date', 'Close']
# data.to_csv("aapl_close.csv", index=False)

# # Setup
# setup(data=data, target='Close', session_id=123, fold=3)

# # Compare models
# best_model = compare_models(sort='R2')

# # 🔁 Pull and save comparison table
# compare_df = pull()
# compare_df.to_csv("ts_model_comparison.csv", index=False)

# # Finalize and save best model
# final_model = finalize_model(best_model)
# save_model(final_model, 'aapl_best_model')
from pycaret.classification import (
    setup, compare_models, finalize_model, save_model, pull,
    create_model, plot_model
)
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
df['target'] = iris.target

# Setup PyCaret classification
setup(data=df, target='target', session_id=123, verbose=False)

# Compare models
best_model = compare_models(sort='Accuracy')

# Save comparison table
comparison_df = pull()
comparison_df.to_csv("model_comparison.csv", index=False)

# Finalize and save best model
final_model = finalize_model(best_model)
save_model(final_model, "model")
lr = create_model('lr')

# Plotar curva AUC
# plot_model(lr, plot='auc', save=True)

plot_model(lr, plot='auc', save=True, plot_kwargs={"multiclass": True})
# Plotar matriz de confusão com porcentagens
plot_model(lr, plot='confusion_matrix', plot_kwargs={'percent': True}, save =True)

plot_model(lr, plot='pr', save=True)

plot_model(lr, plot='feature_all', save=True)


