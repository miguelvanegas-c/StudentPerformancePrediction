# Proyecto PTIA: Predicción del Rendimiento Académico

**Autores:** David Eduardo Salamanca Aguilar, Miguel Ángel Vanegas Cárdenas  
**Curso:** Principios y Tecnologías de Inteligencia Artificial (Grupo 02)  
**Programa:** Ingeniería en Sistemas  
**Fecha:** Septiembre 2025

---

## 📌 Descripción

Este proyecto aplica técnicas de aprendizaje automático para predecir el rendimiento académico de estudiantes de secundaria. Utiliza el conjunto de datos **Student Performance** del repositorio UCI y se enfoca en construir un sistema de alerta temprana que identifique estudiantes en riesgo de bajo desempeño.

---

## 📂 Estructura del Repositorio

PROYECTOPTIA/

├── data/ # Conjuntos de datos originales

├── models/ # Modelo entrenado (Random Forest)

├── reports/ # Informes de métricas y explicabilidad 

│├──images/ # Visualizaciones SHAP y Feature Importance
├── results/ # Predicciones generadas

├── src/ # Código fuente del pipeline ML

│ ├── load_data.py

│ ├── preprocess.py

│ ├── train_model.py

│ ├── evaluate_model.py

│ ├── explain_model.py

│ └── predict.py ├── venv/ # Entorno virtual (ignorado en Git)

---

## 🎯 Objetivo General

Desarrollar y validar un modelo de ML que prediga el rendimiento académico de estudiantes con precisión suficiente para ser útil como sistema de alerta temprana.

---

## 🔎 Metodología

Se sigue el enfoque **CRISP-DM**:

- **Comprensión del problema:** análisis conceptual del bajo rendimiento.
- **Comprensión de los datos:** uso del dataset UCI Student Performance.
- **Preparación de los datos:** limpieza, codificación y normalización.
- **Modelado:** entrenamiento con Random Forest y validación cruzada.
- **Evaluación:** métricas como Accuracy, F1-score y ROC-AUC.
- **Interpretación:** uso de SHAP y Feature Importance para explicabilidad.

---

## 📊 Resultados Destacados

- **Accuracy:** 0.9722
- **ROC_AUC:** 0.9943
- **Recall:** 0.9902
- **Variables clave:** Feature_29 y Feature_28 (calificaciones previas)

---

## 🛠️ Herramientas Utilizadas

- **Lenguaje:** Python
- **Librerías:** pandas, NumPy, scikit-learn, matplotlib, seaborn, SHAP

---

## 🚀 Recomendaciones Futuras

Expandir el sistema hacia una **IA prescriptiva** que recomiende acciones personalizadas para mejorar el rendimiento de estudiantes en riesgo, utilizando análisis contrafactual.

---

## ▶️ Cómo Ejecutar

1. Clonar el repositorio.
2. Crear entorno virtual:
   ```bash
   python -m venv venv
   ```
3. Activar entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar scripts en orden
   - load_data.py
   - preprocess.py
   - train_model.py
   - evaluate_model.py
   - explain_model.py
   - predict.py

## Licencia

Este proyecto es académico y se presenta como parte del curso de Inteligencia Artificial. No está destinado para uso comercial sin autorización.
