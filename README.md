# Spotify
# Proyecto: Predicción de la Popularidad de Canciones en Spotify

## Descripción

Este proyecto aplica técnicas de Machine Learning para predecir la popularidad de una canción en Spotify utilizando únicamente sus características musicales.  
El enfoque combina modelos de clasificación y regresión para abordar de forma eficaz el comportamiento del dataset, el cual contiene muchas canciones con popularidad igual a cero.

## Objetivo

Predecir el nivel de popularidad (`popularity`) de una canción en función de variables como `danceability`, `energy`, `valence`, `tempo`, `acousticness`, entre otras.

## Enfoque del Proyecto

Dado que muchas canciones tienen `popularity = 0` sin haber sido escuchadas o valoradas, se diseñó un pipeline en dos fases:

1. **Clasificación** (`low_popularity`): Detectar si una canción será o no popular.
2. **Regresión** (`popularity`): Predecir cuán popular será una canción que sí tuvo interacción real.

## Dataset

El dataset proviene de Spotify y contiene información musical extraída con la API, junto con valores de popularidad.  
Fue limpiado, procesado y enriquecido con una nueva variable: `low_popularity`, que permite diferenciar entre canciones que fueron escuchadas y las que no.

- Registros: 185,273 canciones
- Variables: 18 (incluyendo numéricas, categóricas y binarias)

## Modelos Utilizados

**Modelos de Clasificación (`low_popularity`):**
- Regresión Logística
- Árbol de Decisión
- Random Forest Classifier (mejor resultado)

**Modelos de Regresión (`popularity`):**
- Regresión Lineal
- Random Forest Regressor (mejor resultado)

## Evaluación

Los modelos fueron evaluados con métricas:

- Clasificación: accuracy, precision, recall, F1-score
- Regresión: MAE, RMSE, R²

El mejor rendimiento se obtuvo con Random Forest en ambos casos:
- Accuracy clasificación: 0.94
- R² en regresión: 0.48

## Estructura del Repositorio

```
ML_Prediccion_Popularidad_Spotify/
│
├── src/
│   ├── data/                  # Dataset limpio final
│   ├── data_sample/           # Muestra del Dataset
│   ├── img/                   # Imágenes generadas por los modelos
│   ├── models/                # Modelos entrenados guardados en .pkl
│   ├── notebooks/             # Notebooks de pruebas
│   └── results_notebook/     # Notebook final: Spotify.ipynb
│
└── README.md
```

## Reproducción del Proyecto

1. Clonar el repositorio
2. Instalar las dependencias necesarias (pandas, scikit-learn, matplotlib, seaborn, joblib, etc.)
3. Ejecutar el notebook `src/results_notebook/Spotify.ipynb` de principio a fin

## Conclusión

El modelo construido permite anticipar si una canción será popular y, si lo es, estimar su nivel de popularidad.  
Este enfoque es útil para plataformas de streaming, análisis de tendencias musicales o herramientas de marketing musical.

## Presentación del Proyecto

[Link del video]


---

# Project: Predicting Song Popularity on Spotify Using Machine Learning

## Description

This project applies Machine Learning techniques to predict the popularity of a song on Spotify using only its musical features.  
The approach combines classification and regression models to effectively handle the dataset, which includes many songs with a popularity score of zero.

## Objective

To predict the `popularity` level of a song based on features such as `danceability`, `energy`, `valence`, `tempo`, `acousticness`, among others.

## Project Approach

Since many songs have `popularity = 0` without being listened to or rated, the pipeline was designed in two stages:

1. **Classification** (`low_popularity`): Detect whether a song is likely to be popular or not.
2. **Regression** (`popularity`): Predict how popular a song will be if it has some level of user interaction.

## Dataset

The dataset comes from Spotify and includes musical attributes retrieved via the API, along with popularity values.  
It was cleaned, processed, and enriched with a new variable: `low_popularity`, which helps distinguish between songs that have been listened to and those that have not.

- Records: 185,273 songs
- Features: 18 (including numerical, categorical, and binary variables)

## Models Used

**Classification Models (`low_popularity`):**
- Logistic Regression
- Decision Tree
- Random Forest Classifier (best performer)

**Regression Models (`popularity`):**
- Linear Regression
- Random Forest Regressor (best performer)

## Evaluation

Models were evaluated using appropriate metrics:

- Classification: accuracy, precision, recall, F1-score
- Regression: MAE, RMSE, R²

The best performance was achieved with Random Forest in both tasks:
- Classification Accuracy: 0.94
- Regression R²: 0.48

## Repository Structure

```
ML_Prediccion_Popularidad_Spotify/
│
├── src/
│   ├── data/                  # Final cleaned dataset
│   ├── data_sample/           # Dataset sample
│   ├── img/                   # Images generated from models
│   ├── models/                # Trained models saved as .pkl
│   ├── notebooks/             # Experimentation notebooks
│   └── results_notebook/     # Final notebook: Spotify.ipynb
│
└── README.md
```

## How to Reproduce

1. Clone the repository
2. Install the necessary dependencies (pandas, scikit-learn, matplotlib, seaborn, joblib, etc.)
3. Run the notebook `src/results_notebook/Spotify.ipynb` from start to finish

## Conclusion

The final model is capable of anticipating whether a song will be popular, and if so, estimating its popularity level.  
This approach can be useful for streaming platforms, trend analysis in music, or marketing support tools.

## Project Presentation

[Video Link]

