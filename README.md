# RePoints -- Proyecto Integrador

Sistema inteligente para identificar residuos de plástico y vidrio con Raspberry Pi e IA. PUCE Manabí, Quinto Semestre, Ingeniería en Software.

## Estructura del proyecto

```
modelo/               Pipeline de datos, entrenamiento y evaluación del modelo de IA
├── Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb   Notebook principal (todo el pipeline)
├── clean_and_split.py                                  Limpieza del dataset + split 70/15/15
├── analyze.py                                           EDA (duplicados, nitidez, brillo)
├── retrain_v2.py, retrain_v3_real.py                    Experimentos de reentrenamiento (ver notas abajo)
├── model/                                                Modelos entrenados (.keras / .tflite)
├── *.json                                                Reportes de resultados (report.json, train_report*.json)
└── figs/, figs_impl/, preview*/, samples/                Figuras e imágenes generadas para los informes

sistema_experto/       Motor de reglas que valida la predicción del modelo antes de decidir
└── sistema_experto.py  Clase SistemaExperto: R0 (calidad de cámara) + R1/R2/R3 (confianza)

dataset/                Dataset original (Kaggle "Drinking Waste Classification")
dataset_split/          Dataset limpio, dividido en train/val/test (generado por clean_and_split.py)
dataset_real/           Fotos reales propias (fuera del dominio de Kaggle), para fine-tuning futuro
├── plastico/            Fotos reales de botellas de plástico
├── vidrio/               (pendiente -- todavía no hay fotos de vidrio real)
└── Plan_Captura_Dataset.xlsx   Checklist para coordinar la sesión de fotos

informes/               Entregables oficiales: los 3 informes (Análisis, Diseño, Implementación),
                         el guion de presentación y el documento de defensa oral (docx + pdf)

rubricas/                Rúbricas de evaluación del proyecto integrador
```

## Modelos entrenados

| Modelo | Descripción | Estado |
|---|---|---|
| `modelo/model/ecosort_mobilenetv2.keras` / `.tflite` | **Modelo oficial**, documentado en los 3 informes. Accuracy test 97.02%. | ✅ Usar este |
| `modelo/model/ecosort_mobilenetv2_v2.keras` / `.tflite` | Experimento: más augmentation de color + más fine-tuning. Mejoró en test set (98.51%) pero empeoró en fotos reales (más confiadamente incorrecto). | ❌ No usar |
| `modelo/model/ecosort_mobilenetv2_v3_real.keras` | Experimento: 10 fotos reales integradas al train set. No mejoró (accuracy test bajó a 96.77%, fotos reales siguieron mal clasificadas). | ❌ No usar |

Los experimentos v2/v3 quedan documentados como evidencia de qué se probó y por qué no funcionó (ver `modelo/train_report_v2.json` y `modelo/train_report_v3_real.json`) -- útil para la sección de "trabajo futuro" de la defensa.

## Cómo correr el pipeline

```bash
python -m pip install -r requirements.txt
python modelo/clean_and_split.py      # limpia dataset/ y genera dataset_split/
# luego correr modelo/Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb con Jupyter
```

## Sistema experto

```bash
python sistema_experto/sistema_experto.py   # corre la demo con casos de ejemplo
```

## Pendientes conocidos

- `dataset_real/vidrio/` sin fotos reales todavía (limitación activa).
- Integrar dataset externo de Kaggle (`mostafaabla/garbage-classification`, clases plastic/green-glass/brown-glass/white-glass) para variedad de color de vidrio -- pendiente de descarga manual (requiere login de Kaggle).
- Fine-tuning dedicado con dataset real ampliado, usando la estrategia correcta (fase final separada, no mezclada con Kaggle) una vez haya suficiente volumen de fotos propias.
