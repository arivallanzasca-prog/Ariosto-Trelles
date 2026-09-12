# Ariosto-Trelles

Apomediación algorítmica y desinformación sobre salud mental en YouTube:
Desarrollo y validación de un modelo XGBoost basado en los criterios DISCERN y HONcode.

## Descripción

Pipeline en R para el análisis automatizado de videos de YouTube en español sobre salud mental.
Incluye:

- Extracción de metadatos vía YouTube Data API v3.
- Evaluación con DISCERN (16 criterios) y HONcode (8 principios).
- Modelo XGBoost optimizado para alta sensibilidad.
- Red de similitud semántica entre videos (TF-IDF + coseno + kNN + Louvain).
- Generación de tablas y figuras reproducibles.

## Estructura

```
R/                  Scripts del pipeline
data/raw/           Datos crudos
data/processed/     Datos procesados
models/             Modelo XGBoost y metadatos
outputs/figures/    Figuras generadas
outputs/tables/     Tablas CSV generadas
docs/               Documentación
```

## Instalación

```r
renv::restore()
shiny::runApp('app.R')
```

## Configuración

Copia `.Renviron.example` como `.Renviron` y rellena tus credenciales:

```
YOUTUBE_CLIENT_ID=tu_client_id
YOUTUBE_CLIENT_SECRET=tu_client_secret
```

## Reproducibilidad

```r
source('R/99_run_all.R')
```

## Licencia

MIT

## Contacto

Autor de correspondencia: [tu correo]
