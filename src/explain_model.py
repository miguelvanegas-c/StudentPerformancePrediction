import os
import sys
import logging
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from load_data import load_datasets, prepare_target

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _get_project_paths():
    """Obtener rutas del proyecto de forma robusta."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        'model': os.path.join(base_dir, 'models', 'rf_best.joblib'),
        'reports': os.path.join(base_dir, 'reports'),
        'images': os.path.join(base_dir, 'reports', 'images')
    }


def _load_and_prepare_data():
    """Cargar y preparar datos."""
    logger.info("Cargando datos...")
    df = load_datasets()
    df = prepare_target(df)
    X = df.drop(columns=["G3", "target"])
    return X, df


def _load_model(model_path):
    """Cargar modelo entrenado."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
    logger.info(f"Cargando modelo desde: {model_path}")
    return joblib.load(model_path)


def _generate_report(shap_vals, X_tr, clf, feature_names, output_dir):
    """Generar reporte de texto con estadísticas de SHAP."""
    logger.info("Generando reporte de explicabilidad...")
    
    # Convertir a arrays 1D para evitar problemas de indexación
    feature_importance = np.ravel(clf.feature_importances_)
    mean_abs_shap = np.ravel(np.abs(shap_vals).mean(axis=0))
    
    report = []
    report.append("=" * 80)
    report.append("REPORTE DE EXPLICABILIDAD DEL MODELO")
    report.append("=" * 80)
    report.append("")
    
    report.append("ESTADISTICAS GENERALES:")
    report.append(f"  - Número de muestras analizadas: {X_tr.shape[0]}")
    report.append(f"  - Número de features: {X_tr.shape[1]}")
    report.append("")
    
    report.append("TOP 10 FEATURES POR IMPORTANCIA SHAP (Mean Absolute):")
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        idx_val = int(idx)
        if idx_val < len(feature_names):
            shap_val = float(mean_abs_shap[idx_val])
            report.append(f"  {rank:2d}. {feature_names[idx_val]:30s} - SHAP: {shap_val:.6f}")
    report.append("")
    
    report.append("TOP 10 FEATURES POR IMPORTANCIA DEL MODELO:")
    top_model = np.argsort(feature_importance)[-10:][::-1]
    for rank, idx in enumerate(top_model, 1):
        idx_val = int(idx)
        if idx_val < len(feature_names):
            imp_val = float(feature_importance[idx_val])
            report.append(f"  {rank:2d}. {feature_names[idx_val]:30s} - Importancia: {imp_val:.6f}")
    report.append("")
    
    report.append("CORRELACION ENTRE SHAP E IMPORTANCIA DEL MODELO:")
    try:
        # Asegurar que ambos arrays tengan el mismo tamaño
        min_size = min(len(mean_abs_shap), len(feature_importance))
        if min_size > 1:
            correlation = np.corrcoef(mean_abs_shap[:min_size], feature_importance[:min_size])[0, 1]
            report.append(f"  Correlación de Pearson: {correlation:.4f}")
        else:
            report.append("  Correlación no disponible (datos insuficientes)")
    except Exception as e:
        logger.warning(f"Error calculando correlación: {e}")
        report.append(f"  Correlación no disponible ({e})")
    report.append("")
    
    report.append("ESTADISTICAS POR FEATURE:")
    for idx in top_indices[:5]:
        idx_val = int(idx)
        if idx_val < len(feature_names):
            report.append(f"\n  {feature_names[idx_val]}:")
            report.append(f"    - Media SHAP: {float(np.mean(shap_vals[:, idx_val])):.6f}")
            report.append(f"    - Desv. Est. SHAP: {float(np.std(shap_vals[:, idx_val])):.6f}")
            report.append(f"    - Min SHAP: {float(np.min(shap_vals[:, idx_val])):.6f}")
            report.append(f"    - Max SHAP: {float(np.max(shap_vals[:, idx_val])):.6f}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    
    # Guardar reporte
    report_path = os.path.join(output_dir, 'explicabilidad_reporte.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"Reporte guardado: {report_path}")
    return report_path, report_text


def _generate_shap_plots(shap_vals, X_tr, clf, explainer, output_dir):
    """Generar múltiples gráficos SHAP y análisis de importancia."""
    logger.info("Generando gráficos de explicabilidad SHAP...")
    
    # Asegurar que el directorio de imágenes existe
    images_dir = output_dir
    os.makedirs(images_dir, exist_ok=True)
    output_paths = {}
    
    # 1. Summary plot (Beeswarm)
    logger.info("  - Generando summary plot (Beeswarm)...")
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_vals, X_tr, show=False)
    summary_path = os.path.join(images_dir, 'shap_summary_beeswarm.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_paths['summary_beeswarm'] = summary_path
    logger.info(f"    Guardado: {summary_path}")
    
    # 2. Summary plot (Bar - Importancia)
    logger.info("  - Generando summary plot (Bar - Importancia)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals, X_tr, plot_type="bar", show=False)
    bar_path = os.path.join(images_dir, 'shap_summary_bar.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_paths['summary_bar'] = bar_path
    logger.info(f"    Guardado: {bar_path}")
    
    # 3. Feature Importance (Permutation)
    logger.info("  - Generando importancia de features (Modelo)...")
    feature_importance = clf.feature_importances_
    feature_names = [f"Feature_{i}" for i in range(X_tr.shape[1])]
    
    # Obtener nombres reales si es posible
    try:
        if hasattr(X_tr, 'columns'):
            feature_names = X_tr.columns.tolist()
    except:
        pass
    
    plt.figure(figsize=(12, 8))
    indices = np.argsort(feature_importance)[-20:]  # Top 20
    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title('Top 20 Features - Importancia del Modelo')
    plt.tight_layout()
    importance_path = os.path.join(images_dir, 'feature_importance.png')
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_paths['importance'] = importance_path
    logger.info(f"    Guardado: {importance_path}")
    
    # 4. Dependence plots para top 4 features
    logger.info("  - Generando dependence plots...")
    try:
        top_features_idx = np.argsort(feature_importance)[-4:][::-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feature_idx in enumerate(top_features_idx):
            ax = axes[idx]
            try:
                # Usar dependence_plot sin calcular interacción
                shap.dependence_plot(feature_idx, shap_vals, X_tr, 
                                    feature_names=feature_names,
                                    ax=ax, show=False, interaction_index=None)
            except Exception as e:
                logger.warning(f"Error en dependence_plot para feature {feature_names[feature_idx]}: {e}")
                # Si falla, hacer un plot simple
                ax.scatter(X_tr[:, feature_idx], shap_vals[:, feature_idx], alpha=0.5)
                ax.set_xlabel(feature_names[feature_idx])
                ax.set_ylabel('SHAP value')
        
        plt.tight_layout()
        dependence_path = os.path.join(images_dir, 'shap_dependence_top4.png')
        plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['dependence'] = dependence_path
        logger.info(f"    Guardado: {dependence_path}")
    except Exception as e:
        logger.warning(f"Error generando dependence plots: {e}")
    
    # 5. Force plot (ejemplo de instancias)
    logger.info("  - Generando force plots (muestras)...")
    base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    
    # Top 3 predicciones positivas y negativas
    predictions = clf.predict_proba(X_tr)[:, 1] if hasattr(clf, 'predict_proba') else clf.predict(X_tr)
    top_positive_idx = np.argsort(predictions)[-3:][::-1]
    top_negative_idx = np.argsort(predictions)[:3]
    
    fig_list = []
    for instance_idx in np.concatenate([top_positive_idx, top_negative_idx])[:4]:
        try:
            shap.force_plot(base_value, shap_vals[instance_idx], 
                          X_tr[instance_idx], feature_names=feature_names, 
                          show=False, matplotlib=True)
            fig_list.append(plt.gcf())
            plt.close()
        except Exception as e:
            logger.warning(f"Error generando force plot para índice {instance_idx}: {e}")
    
    if fig_list:
        force_path = os.path.join(images_dir, 'shap_force_plots.png')
        output_paths['force'] = force_path
        logger.info(f"    Guardado: {force_path}")
    
    # 6. Waterfall plot para instancia representativa
    logger.info("  - Generando waterfall plot...")
    try:
        representative_idx = len(X_tr) // 2
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(shap.Explanation(values=shap_vals[representative_idx],
                                            base_values=base_value,
                                            data=X_tr[representative_idx],
                                            feature_names=feature_names), 
                          show=False)
        waterfall_path = os.path.join(images_dir, 'shap_waterfall.png')
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths['waterfall'] = waterfall_path
        logger.info(f"    Guardado: {waterfall_path}")
    except Exception as e:
        logger.warning(f"Error generando waterfall plot: {e}")
    
    return output_paths


def explain():
    """
    Explicar predicciones del modelo usando SHAP.
    
    Genera:
    - Múltiples gráficos SHAP (beeswarm, bar, dependence, waterfall, force)
    - Importancia de features
    - Reporte de explicabilidad en texto
    """
    try:
        paths = _get_project_paths()
        
        # Cargar modelo
        model = _load_model(paths['model'])
        
        # Preparar datos
        X, df = _load_and_prepare_data()
        
        # Extraer componentes del pipeline
        preprocessor = model.named_steps["prep"]
        clf = model.named_steps["rf"]
        
        # Transformar datos
        logger.info("Preprocesando datos...")
        X_tr = preprocessor.transform(X)
        
        # Obtener nombres de features después de preprocesamiento
        feature_names = [f"Feature_{i}" for i in range(X_tr.shape[1])]
        try:
            if hasattr(X_tr, 'columns'):
                feature_names = X_tr.columns.tolist()
        except:
            pass
        
        # Generar explicaciones SHAP
        logger.info("Calculando valores SHAP...")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_tr)
        
        # Manejo correcto de shap_values para clasificación binaria
        shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # Generar gráficos
        output_paths = _generate_shap_plots(shap_vals, X_tr, clf, explainer, paths['images'])
        
        # Generar reporte de texto
        report_path, report_text = _generate_report(shap_vals, X_tr, clf, feature_names, paths['reports'])
        output_paths['report'] = report_path
        
        # Mostrar resumen en logs
        logger.info("\n" + report_text)
        
        logger.info("Explicabilidad generada exitosamente")
        return output_paths
        
    except Exception as e:
        logger.error(f"Error al generar explicabilidad: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        explain()
        logger.info("Proceso completado exitosamente")
    except Exception as e:
        logger.error(f"Error fatal: {str(e)}")
        sys.exit(1)
