# Aprendizaje Automático - Modelos Avanzados



 **Fork de un proyecto académico en equipo (2 personas)** — Aprendizaje
 Automático, UC3M.
 Original: https://github.com/100522320/g81_P1_03_Aprendizaje_Automatico



# g81_P1_03 — Clasificación de Clientes Bancarios con ML

Pipeline de clasificación supervisada sobre datos de clientes bancarios:
EDA, entrenamiento de modelos y despliegue en una app interactiva.

## Contenido
- `notebook_final.ipynb` — pipeline completo (EDA, modelos, evaluación)
- `mystreamlit.py` — app interactiva de predicciones
- `modelo_final.pkl` — modelo final entrenado
- `predicciones.csv` — predicciones sobre el dataset de test
- `requirements.txt` — dependencias

## Uso
```bash
pip install -r requirements.txt
python -m streamlit run mystreamlit.py
```

## Enfoque técnico
Random Forest, SVM y Regresión Logística, con optimización de
hiperparámetros (`GridSearchCV`) y evaluación por validación cruzada
(Accuracy, F1-Score, ROC-AUC).

