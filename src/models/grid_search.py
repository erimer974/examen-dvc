import pandas as pd
import yaml
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Chargement des paramètres
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')

ridge = Ridge()
grid = GridSearchCV(
    estimator=ridge, 
    param_grid=params['gridsearch']['parameters'], 
    cv=params['gridsearch']['cv']
)
grid.fit(X_train_scaled, y_train)
best_params = grid.best_params_

# Sauvegarde des meilleurs paramètres dans un fichier pickle
with open('./models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)