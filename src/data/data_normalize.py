import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Chargement des données
X_train = pd.read_csv(r'./data/processed_data/X_train.csv')
X_test = pd.read_csv(r'./data/processed_data/X_test.csv')

# Initialisation et apprentissage du scaler sur les données numériques d'entraînement
scaler = MinMaxScaler()
scaler.fit(X_train)

# Transformation des données
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversion en DataFrame avec les colonnes d'origine
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Sauvegarde des données normalisées
X_train_scaled.to_csv(r'./data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv(r'./data/processed_data/X_test_scaled.csv', index=False)

