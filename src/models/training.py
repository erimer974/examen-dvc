import pickle
import yaml
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des paramètres
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Chargement des données
X_train_scaled = pd.read_csv('data/processed_data/X_train_scaled.csv')
X_test_scaled = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

with open('./models/best_params.pkl', 'rb') as f:
    best_params_loaded = pickle.load(f)

# Vérifier le type de modèle à utiliser
model_type = params['training']['model_type']
if model_type == 'ridge':
    # Création d'un modèle Ridge avec ces paramètres
    best_model = Ridge(**best_params_loaded)
else:
    # Par défaut, utiliser Ridge
    best_model = Ridge(**best_params_loaded)

best_model.fit(X_train_scaled, y_train)

# Sauvegarde du modèle entraîné dans un fichier pickle
with open('./models/gbr_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)