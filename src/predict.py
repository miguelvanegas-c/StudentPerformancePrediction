import os
import sys
import joblib
import pandas as pd
from load_data import load_datasets, prepare_target


def _get_project_paths():
    """Obtener rutas del proyecto de forma robusta."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'model': os.path.join(base_dir, 'models', 'rf_best.joblib'),
        'results': os.path.join(base_dir, 'results')
    }


def generate_predictions():
    """Generar predicciones usando el modelo entrenado."""
    
    try:
        paths = _get_project_paths()
        
        # Verificar que la carpeta results exista
        os.makedirs(paths['results'], exist_ok=True)

        # Cargar modelo entrenado
        print("Cargando modelo...")
        if not os.path.exists(paths['model']):
            raise FileNotFoundError(f"Modelo no encontrado en: {paths['model']}")
        model = joblib.load(paths['model'])

        # Cargar datos
        print("Cargando datos...")
        df = load_datasets()
        df = prepare_target(df)

        # Separar X
        X = df.drop(columns=["G3", "target"])

        # Hacer predicciones
        print("Generando predicciones...")
        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Construir DataFrame de resultados
        results_df = pd.DataFrame({
            "prob_aprobar": probs,
            "prediccion": preds
        })

        # Guardar archivo en results/
        output_path = os.path.join(paths['results'], 'predictions.csv')
        results_df.to_csv(output_path, index=False)

        print(f"Predicciones guardadas en {output_path}")
        
        # Mostrar estadísticas
        print(f"\nEstadísticas de predicciones:")
        print(f"  - Total muestras: {len(results_df)}")
        print(f"  - Predicción: Aprobar (1): {(preds == 1).sum()}, No aprobar (0): {(preds == 0).sum()}")
        print(f"  - Probabilidad promedio: {probs.mean():.4f}")
        
        return output_path
        
    except Exception as e:
        print(f"Error al generar predicciones: {e}")
        raise


if __name__ == "__main__":
    generate_predictions()
