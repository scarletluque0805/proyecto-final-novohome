# Predicción de devoluciones y reclamos — Novohome en Mercado Libre

Proyecto final de **Data Science II: Machine Learning para la Ciencia de Datos** (Coderhouse) · Autora: **Elizabeth Luque**

## Problema
Novohome vende artículos para el hogar, pequeños electrodomésticos, grifería y belleza en Mercado Libre. Cada
devolución o reclamo genera costos logísticos y afecta la reputación de la cuenta. El objetivo es estimar, **al momento
de la venta**, la probabilidad de que una orden termine en devolución o reclamo, para actuar antes (confirmar
compatibilidad, enviar instrucciones, mejorar la publicación).

| | |
|---|---|
| Tarea | Clasificación binaria |
| Variable objetivo | `devolucion_o_reclamo` |
| Métricas | ROC-AUC y PR-AUC (clase minoritaria, ~7%) + **ahorro neto en ARS** |
| Datos | Sintéticos, sin información real de clientes ni productos (ver `data/README.md`) |

## Resultados
| Modelo | ROC-AUC validación | Ahorro validación |
|---|---|---|
| Regla simple (grifería o entrega > 5 días) | 0,705 | — |
| Regresión logística | 0,795 | $83.100 |
| Random Forest | 0,764 | $66.300 |
| **XGBoost optimizado (elegido)** | **0,795** | **$84.900** |

En **test** (abril–junio 2026, nunca usado antes): ROC-AUC **0,721**. Con umbral 0,68 se contacta el 10% de las órdenes
y se alcanza el 34% de las devoluciones, con ahorro neto positivo.
Variables más influyentes según **SHAP**: días de entrega, puntaje de la ficha, categoría grifería y plomería, precio,
logística por Colecta y descuento.

## Estructura
```
proyecto-final-novohome/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── proyecto_final_novohome.ipynb   # Pipeline completo, ejecutado y con salidas
├── src/
│   └── generar_datos.py                # Genera los datos sintéticos y los carga en SQLite
├── data/README.md                      # Descripción de los datos (la base se genera, no se versiona)
├── reports/figures/                    # Gráficos del EDA, curvas y SHAP
└── models/                             # Modelo entrenado (se genera al ejecutar)
```

## Cómo reproducir
```bash
git clone <URL-de-este-repositorio>
cd proyecto-final-novohome
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_final_novohome.ipynb   # Ejecutar todas las celdas
```
**En Google Colab:** abrir una celda y ejecutar
`!git clone <URL-de-este-repositorio> && cd proyecto-final-novohome && pip install -r requirements.txt`,
luego abrir el notebook desde la carpeta clonada (el notebook detecta si se ejecuta desde `notebooks/` o desde la raíz).

Todo usa semilla fija (42): la última celda verifica con 11 controles automáticos (esperado vs observado) que los
datos, el split, el modelo guardado y los resultados sean reproducibles.

## Pipeline
1. **Ingesta:** generación de datos y carga en SQLite con transacción explícita; extracción con un JOIN SQL.
2. **EDA:** nulos, duplicados, outliers (IQR), tipo de faltante, distribuciones y correlaciones.
3. **Limpieza e ingeniería de atributos:** se elimina la columna con fuga de información, se normalizan textos, se corrigen precios erróneos y se crean 8 variables nuevas.
4. **Split temporal:** train 2025, validación 2026-T1, test 2026-T2.
5. **Modelos:** regresión logística, Random Forest y XGBoost en `Pipeline` + `ColumnTransformer` (imputación, escalado y one-hot ajustados solo con train), validación cruzada estratificada de 5 folds.
6. **Optimización:** `RandomizedSearchCV` (20 combinaciones) sobre XGBoost.
7. **Selección:** umbral que maximiza el ahorro neto en validación.
8. **Evaluación final en test, SHAP** (importancia global y explicación de una orden) y guardado con `joblib`.

## Limitaciones
Datos sintéticos y costos de negocio supuestos (devolución $18.000, contacto $1.500, 35% de efectividad; editables
al inicio del notebook). Con datos reales habría que recalibrar el umbral, monitorear el desempeño en el tiempo y
medir la efectividad del contacto preventivo con una prueba A/B.
