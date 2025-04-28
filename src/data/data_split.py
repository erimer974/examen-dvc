import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_params(params_file='params.yaml'):
    """Charge les paramètres depuis le fichier yaml"""
    with open(params_file, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_data():
    """Charge les données depuis l'URL"""
    df = pd.read_csv('https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv')
    return df

def preprocess(df):
    """Prétraite les données"""
    # Conversion et distribution date
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    df['Day'] = df['date'].dt.day
    df = df.drop(['date'], axis=1)
    return df

def split_data(df, params):
    """Divise les données en ensembles d'entraînement et de test selon les paramètres"""
    test_size = params['split']['test_size']
    random_state = params['split']['random_state']
    
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    
    # Chargement des paramètres
    params = load_params()
    
    # Étape 1: Chargement des données depuis l'URL
    df = load_data()
    
    # Étape 2: Sauvegarde des données brutes
    df.to_csv(r'./data/raw_data/raw.csv', index=False)
    
    # Étape 3: Prétraitement des données
    df = preprocess(df)

    # Étape 4: Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = split_data(df, params)

    # Étape 5: Sauvegarde des données
    df.to_csv(r'./data/processed_data/processed.csv', index=False)
    X_train.to_csv(r'./data/processed_data/X_train.csv', index=False)
    X_test.to_csv(r'./data/processed_data/X_test.csv', index=False)
    y_train.to_csv(r'./data/processed_data/y_train.csv', index=False)
    y_test.to_csv(r'./data/processed_data/y_test.csv', index=False)
 