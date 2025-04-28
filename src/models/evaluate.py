import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import json
from pathlib import Path

def main(repo_path):
    # Construction des chemins
    data_path = repo_path / "data/processed_data"
    model_path = repo_path / "models/gbr_model.pkl"
    metrics_dir = repo_path / "metrics"
    predictions_csv_path = metrics_dir / "predictions.csv"
    metrics_json_path = metrics_dir / "scores.json"

    # Chargement des données de test
    X_test_scaled = pd.read_csv(data_path / "X_test_scaled.csv")
    y_test = pd.read_csv(data_path / "y_test.csv")
    y_test_values = y_test.values.flatten()

    # Chargement du modèle
    with open(model_path, "rb") as f:
        best_model_loaded = pickle.load(f)

    # Prédictions
    y_pred = best_model_loaded.predict(X_test_scaled)
    y_pred = np.array(y_pred).flatten()

    # Création et sauvegarde du DataFrame des prédictions
    df_predictions = pd.DataFrame({"y_true": y_test_values, "y_pred": y_pred})
    df_predictions.to_csv(predictions_csv_path, index=False)

    # Calcul des métriques
    mse = mean_squared_error(y_test_values, y_pred)
    r2 = r2_score(y_test_values, y_pred)

    # Sauvegarde des métriques dans un fichier JSON
    metrics = {"mse": mse, "r2": r2}
    metrics_json_path.write_text(json.dumps(metrics))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent.resolve()
    main(repo_path)
