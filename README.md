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

Se evaluó un conjunto amplio de algoritmos de clasificación para predecir la
suscripción de clientes a un producto bancario, comparándolos de forma
progresiva:

- **Regresión Logística (sin y con regularización L1):** modelos de partida
  para establecer una línea base simple e interpretable antes de probar
  algoritmos más complejos.
- **SVM (kernel RBF):** se eligió el kernel RBF porque permite proyectar los
  datos a un espacio de dimensiones infinitas, encontrando siempre un
  hiperplano que separe las clases aunque estén mezcladas en el espacio
  original.
- **K-Nearest Neighbors (KNN):** incluido como referencia adicional de un
  modelo no paramétrico basado en la cercanía entre observaciones.
- **Árbol de Decisión (Decision Tree):** para tener un modelo base
  interpretable de tipo árbol antes de pasar a métodos de ensemble.
- **Random Forest:** como primer método de ensemble (Bagging), para reducir
  la varianza respecto a un único árbol de decisión.
- **CatBoost:** probado junto a Random Forest y HistGradientBoosting por
  considerarse que, dada la naturaleza del dataset, un método de Boosting
  podía mejorar los resultados de los modelos ya evaluados.
- **HistGradientBoosting:** evaluado por la misma razón que CatBoost;
  finalmente seleccionado como modelo ganador. Con un accuracy prácticamente
  idéntico a CatBoost (0.8593 frente a 0.8594), se prefirió por su eficiencia
  computacional: alcanza el mismo poder predictivo en mucho menos tiempo de
  entrenamiento, gracias a que discretiza los datos en histogramas en lugar
  de evaluar cada valor individual para encontrar el punto de corte óptimo
  en cada nodo.

Todos los modelos (salvo las líneas base iniciales) se optimizaron mediante
búsqueda de hiperparámetros con **Optuna** (`OptunaSearchCV`), y se
compararon con validación cruzada usando **Accuracy** como métrica principal
y análisis de la matriz de confusión para evaluar el impacto de negocio
(Recall sobre la clase minoritaria).

**Conclusión:** la familia de modelos de Gradient Boosting superó de forma
consistente a los enfoques basados en Bagging (Random Forest) para este
dataset. Entre los dos finalistas de Boosting, HistGradientBoosting se
impuso por su eficiencia, siendo el modelo elegido para producción.

