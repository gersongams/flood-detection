# Detector de Riesgo de Inundaciones

Sistema de análisis de imágenes satelitales para identificar y predecir zonas propensas a inundaciones utilizando técnicas de deep learning.

![Mapa de Riesgo de Inundación](./images/03-flood-risk-map.png)

## Descripción

Este proyecto es parte de una tesis de maestría que utiliza imágenes del satélite Sentinel-2 para detectar áreas con alto riesgo de inundación. El sistema combina:

- **Índices espectrales** (NDVI, NDWI) para detectar vegetación y cuerpos de agua
- **Datos de elevación** (DEM) para análisis topográfico
- **Redes neuronales convolucionales** (CNN) para predicción de riesgo

### Tipo de problema

**Clasificación binaria por píxel (segmentación semántica):** para cada píxel de 30 m se predice la probabilidad de que sea zona inundable (`1`) o no (`0`). Por eso las métricas del proyecto son las de clasificación con clases desbalanceadas: **F1 de la clase positiva, Recall y PR-AUC** (ROC-AUC e IoU como complemento). No se usa *accuracy* como métrica principal: los píxeles inundables son minoría y un modelo que siempre predice "no inundable" tendría un *accuracy* alto sin servir de nada.

## Inicio rápido

```bash
uv sync                          # 1. Instalar dependencias
cp .env.example .env             # 2. Configurar credenciales (editar SH_CLIENT_ID / SH_CLIENT_SECRET)
make worker                      # 3. Terminal 1: levanta Redis (Docker) + worker de Celery
make run                         # 4. Terminal 2: levanta la app Streamlit
```

Abrir `http://localhost:8501`. Para detener todo: `make stop`.

## Requisitos

- Python 3.12 (fijado en `.python-version`; el proyecto acepta ≥ 3.11)
- [uv](https://docs.astral.sh/uv/) como gestor de paquetes
- Docker Desktop (para Redis)
- Cuenta en [Sentinel Hub](https://apps.sentinel-hub.com) con credenciales OAuth

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/gersongams/flood-detection.git
cd flood-detection
```

### 2. Instalar dependencias
```bash
uv sync
```
Las versiones exactas están fijadas en `uv.lock`, por lo que `uv sync` reproduce el mismo entorno en cualquier máquina.

### 3. Configurar variables de entorno
```bash
cp .env.example .env
```

Editar `.env` con tus credenciales de Sentinel Hub:
```env
SH_CLIENT_ID=tu_client_id
SH_CLIENT_SECRET=tu_client_secret
REDIS_URL=redis://localhost:6379/0
DATA_DIR=./data
```

> La aplicación no arranca si `SH_CLIENT_ID` o `SH_CLIENT_SECRET` están vacíos (`app/config.py` lanza `ValueError`).

### 4. Iniciar Redis
```bash
docker compose up -d
```

## Uso

### 1. Iniciar el worker de Celery
```bash
uv run celery -A app.celery_app worker -l info --pool=solo
```
`--pool=solo` evita `fork()`, que rompe la aceleración MPS (GPU de Apple Silicon). Equivale a `make worker`.

### 2. Iniciar la aplicación
```bash
uv run streamlit run app/main.py
```

### 3. Abrir en el navegador
Navegar a `http://localhost:8501`

### Notebook de análisis
```bash
uv run jupyter notebook analysis.ipynb
```

### Comandos `make`

| Comando | Acción |
|---------|--------|
| `make install` | `uv sync` |
| `make redis` | Levanta Redis con Docker Compose |
| `make worker` | Redis + worker de Celery (terminal 1) |
| `make run` | Redis + app Streamlit (terminal 2) |
| `make stop` | Detiene Streamlit, Celery y Redis |
| `make clean` | Borra `data/images/*`, `data/models/*` y cachés |

## Flujo de Trabajo

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Seleccionar │     │  2. Descargar   │     │  3. Entrenar    │     │  4. Ver         │
│     Área        │ ──▶ │     Imágenes    │ ──▶ │     Modelo      │ ──▶ │     Resultados  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Seleccionar Área**: Dibujar un rectángulo en el mapa interactivo
2. **Descargar Imágenes**: El sistema descarga imágenes satelitales del período seleccionado
3. **Entrenar Modelo**: La CNN aprende patrones de inundación de los datos
4. **Ver Resultados**: Mapa de riesgo interactivo con estadísticas

## Datos

| Parámetro | Valor | Dónde se define |
|-----------|-------|-----------------|
| Colección | Sentinel-2 L2A (corrección atmosférica) | `app/utils/satellite.py` |
| Resolución | 30 m | `app/config.py` (`RESOLUTION`) |
| Cobertura de nubes máx. | 25 % (mosaico `LEAST_CC`) | `app/config.py` (`MAX_CLOUD_COVERAGE`) |
| Ventana temporal | 5 años | `app/config.py` (`YEARS_BACK`) |
| Imágenes por mes | 4 | `app/config.py` (`IMAGES_PER_MONTH`) |
| Bandas | B02, B03, B04, B08 | ver tabla de bandas |
| Terreno | DEM, pendiente, TWI | `app/pages/1_DEM_Test.py`, `app/tasks.py` |

Las imágenes descargadas se guardan en `data/images/<job_id>/` (con una caché global para no descargar dos veces la misma fecha) y los modelos en `data/models/<job_id>/`. Ambas carpetas están fuera del control de versiones.

## Características

### 1. Descarga de Imágenes Satelitales
Descarga automática de imágenes Sentinel-2 desde Sentinel Hub con:
- Intervalos diarios, semanales o mensuales
- Filtro de cobertura de nubes
- Sistema de caché para evitar descargas duplicadas

### 2. Índices Espectrales
![Índices Espectrales](./images/04-spectral-indices.png)

- **NDVI** (Índice de Vegetación de Diferencia Normalizada): Evalúa la salud de la vegetación
- **NDWI** (Índice de Agua de Diferencia Normalizada): Detecta cuerpos de agua

### 3. Variabilidad Temporal
![Variabilidad Temporal](./images/05-temporal-variability.png)

Análisis de cambios en el tiempo: las áreas con alta variabilidad pueden indicar zonas de inundación histórica.

### 4. Análisis de Terreno (DEM)
![Datos de Terreno](./images/06-terrain-data-dem.png)

- **Elevación**: Áreas bajas tienen mayor riesgo
- **Pendiente**: Terrenos planos acumulan agua
- **TWI** (Índice Topográfico de Humedad): Indica dónde se acumula el agua
- **Distancia al agua**: Proximidad a ríos y cuerpos de agua

![Susceptibilidad Compuesta](./images/07-distance-susceptibility.png)

### 5. Modelo de Deep Learning
![Curva de Entrenamiento](./images/02-training-loss-curve.png)

Red neuronal convolucional (CNN) encoder-decoder (`app/models/flood_model.py`) que aprende patrones de inundación a partir de:
- Series temporales de NDVI y NDWI (2 canales por fecha)
- Características del terreno (elevación, pendiente, TWI)

Configuración actual de entrenamiento: parches de 128×128 con paso de 64, `BCELoss`, Adam (lr = 1e-3) con `OneCycleLR`, 50 épocas, *batch* 16, *early stopping* con paciencia 5.

### 6. Validación del Modelo
![Análisis de Validación](./images/08-proxy-validation.png)

Correlación entre las predicciones del modelo y factores del terreno conocidos.

![Comparación Visual](./images/09-visual-comparison.png)

![Comparación con Terreno](./images/10-terrain-comparison.png)

### 7. Análisis Estadístico
![Análisis Estadístico](./images/11-statistical-analysis.png)

Comparación de valores medios entre zonas de alto y bajo riesgo para validar que el modelo captura correctamente los indicadores de inundación.

## Limitaciones conocidas

Estas limitaciones están identificadas y son el foco del siguiente sprint (ver [Plan: EDA + Baseline](#plan-eda--baseline)).

1. **Las etiquetas son pseudo-etiquetas derivadas de las mismas entradas (leakage).** En `prepare_training_data` (`app/models/flood_model.py`) la etiqueta `y` se calcula con umbrales sobre NDWI/NDVI (`NDWI > 0.3` y `NDVI < 0.2`) más una combinación ponderada de DEM, pendiente y TWI, que son exactamente los canales de entrada `X`. La CNN aprende a reproducir esa fórmula, no a detectar inundaciones. Por eso la validación actual (secciones 6 y 7) es indirecta (*proxy*): mide coherencia con el terreno, no acierto frente a una verdad de campo.
2. **No hay partición train/validación/test.** Todos los parches se usan para entrenar y la única métrica registrada es la pérdida de entrenamiento.
3. **No hay semilla fija.** Dos entrenamientos con los mismos datos pueden dar resultados distintos.
4. **Los parches se solapan al 50 %.** Una partición aleatoria por parche filtraría píxeles entre train y test; la partición tiene que ser espacial (por bloques o por escena).

## Reproducibilidad

| Elemento | Estado actual | Ubicación |
|----------|---------------|-----------|
| Entorno | ✅ fijado | `uv.lock`, `.python-version` |
| Comandos de ejecución | ✅ documentados | este README, `Makefile` |
| Configuración de datos | ✅ centralizada | `app/config.py` |
| Hiperparámetros de entrenamiento | ⚠️ en firmas de funciones | `train_model()` en `app/models/flood_model.py` |
| Métricas de entrenamiento | ⚠️ solo pérdida, JSON por job | `data/models/<job_id>/metrics.json` |
| Semilla / partición | ❌ pendiente | — |
| Logs `logs/metrics_*.txt` | ❌ pendiente | — |
| Versionado de datos (fecha/hash/tamaño) | ❌ pendiente | — |

## Plan: EDA + Baseline

Siguiente sprint, siguiendo la guía *EDA + Baseline mínimo según tipo de proyecto* (Proyecto de Investigación 2). Las tareas marcadas `[ ]` aún no existen en el repositorio.

**Protocolo común**
- [ ] `seed = 42` fijada en `numpy`, `torch` y `random`, guardada junto a cada resultado
- [ ] Partición espacial (por bloques o por escena), documentada y guardada en `data/splits/`
- [ ] Logs automáticos en `logs/metrics_<experimento>_<fecha>.txt` (hora, tamaño de datos, pasos, métricas, configuración)
- [ ] Manifiesto de datos `data/MANIFEST.json` (fecha de descarga, bbox, número de escenas, tamaño y hash SHA-256)

**Etiquetas reales**
- [ ] Evaluar contra [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (chips Sentinel-2 con etiquetas manuales de agua/inundación), el mismo benchmark usado por Konapala et al. (2021)

**EDA (`notebooks/01_eda.ipynb`)**
- [ ] Calidad: NaN y no-data por banda, rangos de reflectancia, fechas duplicadas, nubes residuales
- [ ] Distribuciones: histogramas de NDVI/NDWI por clase, balance de clases
- [ ] Relaciones: correlación entre features (NDVI, NDWI, DEM, pendiente, TWI)
- [ ] Riesgos: desbalance, leakage, *drift* estacional, sesgo por vegetación sobre el agua
- [ ] Cierre con 2–3 decisiones accionables

**Baselines (`scripts/baseline.py`)**

| Modelo | Tipo | Propósito |
|--------|------|-----------|
| `DummyClassifier` (más frecuente / estratificado) | trivial | piso mínimo |
| Umbral NDWI > 0 | regla física | baseline de dominio |
| Regresión logística por píxel | modelo simple (obligatorio) | comparación justa con la CNN |
| CNN actual | modelo propuesto | objetivo a superar |

Métricas: F1 (clase positiva), Recall, PR-AUC; ROC-AUC e IoU opcionales; matriz de confusión.

## Estructura del Proyecto

```
.
├── app/
│   ├── main.py              # Aplicación Streamlit
│   ├── celery_app.py        # Configuración de Celery
│   ├── tasks.py             # Tareas de background (descarga, entrenamiento)
│   ├── config.py            # Configuración
│   ├── models/
│   │   └── flood_model.py   # Modelo CNN PyTorch + pseudo-etiquetas
│   ├── pages/
│   │   └── 1_DEM_Test.py    # Página de prueba de datos de terreno
│   └── utils/
│       └── satellite.py     # Descarga e índices espectrales
├── data/                    # (no versionado)
│   ├── images/              # Imágenes descargadas
│   └── models/              # Modelos entrenados + metrics.json
├── docs/papers/             # Notas de la revisión de literatura (23 papers)
├── images/                  # Capturas para este README
├── scripts/
│   └── zotero_index.sh      # Índice de PDFs de la biblioteca Zotero
├── analysis.ipynb           # Notebook de análisis original
├── docker-compose.yml       # Contenedor Redis
├── Makefile                 # Atajos de ejecución
├── pyproject.toml           # Dependencias UV
├── uv.lock                  # Versiones exactas
└── .env.example             # Plantilla de variables de entorno
```

## Tecnologías

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python 3.12 |
| Gestor de paquetes | UV |
| Framework Web | Streamlit |
| Tareas en background | Celery + Redis |
| Datos satelitales | Sentinel Hub API |
| Deep Learning | PyTorch |
| ML clásico / métricas | scikit-learn |
| Visualización | Folium, Matplotlib |

## Bandas Sentinel-2 Utilizadas

| Banda | Nombre | Uso |
|-------|--------|-----|
| B02 | Azul | Visualización RGB |
| B03 | Verde | NDWI, Visualización RGB |
| B04 | Rojo | NDVI, Visualización RGB |
| B08 | Infrarrojo Cercano (NIR) | NDVI, NDWI |

## Fórmulas

### NDVI (Índice de Vegetación)
```
NDVI = (NIR - Rojo) / (NIR + Rojo) = (B08 - B04) / (B08 + B04)
```
- Valores altos (→1): Vegetación densa y saludable
- Valores bajos (→0): Suelo desnudo, agua, o vegetación muerta

### NDWI (Índice de Agua)
```
NDWI = (Verde - NIR) / (Verde + NIR) = (B03 - B08) / (B03 + B08)
```
- Valores altos (>0.3): Presencia de agua
- Valores bajos (<0): Vegetación o suelo
