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
└── sistema_experto.py  Clase SistemaExperto: R0b (origen/resolución) + R0 (calidad de cámara) + R1/R2/R3 (confianza)

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
| `modelo/model/ecosort_mobilenetv2_v4_vidrio.keras` | Experimento: primera vez con fotos reales de vidrio (4, antes 0) + dataset real de plastico ampliado (75, antes 51). Held-out honesto (~20%, nunca visto en train) antes de entrenar. Test set oficial casi igual (97.02%), pero el sesgo se confirma: 15/15 fotos reales de plastico held-out siguieron prediciéndose como vidrio (0% accuracy); la única foto de vidrio held-out sí acertó (trivial, ya estaba sesgado hacia vidrio). | ❌ No usar |

Los experimentos v2/v3/v4 quedan documentados como evidencia de qué se probó y por qué no funcionó (ver `modelo/train_report_v2.json`, `modelo/train_report_v3_real.json`, `modelo/train_report_v4_vidrio.json`) -- útil para la sección de "trabajo futuro" de la defensa. El hallazgo constante en v3 y v4: el modelo tiene un sesgo sistemático a predecir "vidrio" en fotos reales tomadas en la caja de luz/flash (probablemente por brillos y reflejos que coinciden con rasgos del vidrio en el dataset de Kaggle), independiente del material real -- agregar más fotos reales de plástico al train (incluso oversampleadas x8) no lo corrigió.

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

### Hallazgo de seguridad (2026-08-12): fotos falsas/generadas por IA

Se recibieron 2 tandas por WhatsApp (39 fotos en total) que resultaron ser
imágenes generadas por IA o descargadas de internet, no capturas reales:
texto ilegible/inventado en etiquetas (nutricional, sellos), marcas
genéricas falsas ("Vineyard Hill", "Amber Lager", "Cristalina", "Vino del
Sol"), y **resolución inconsistente** con la cámara real del proyecto
(dataset_real siempre sale en 1600x900 o 1280x960 según la sesión; estas
fotos venían en 1448x1086 o en 8 resoluciones distintas entre sí -- algo
imposible si vinieran del mismo dispositivo).

Más grave: el filtro de calidad R0 (nitidez/brillo) que ya existía **no las
detectaba** -- al contrario, las aprobaba todas (nitidez alta, sin ruido de
cámara real) mientras rechazaba fotos reales legítimas por "borrosas". Se
agregó una regla nueva, **R0b**, que valida la resolución contra una lista
blanca de resoluciones de cámara conocidas *antes* de mirar nitidez/brillo o
el modelo (ver `SistemaExperto.evaluar_calidad` /
`resoluciones_permitidas`). Resultado: 39/40 fotos falsas rechazadas, 0
falsos positivos contra las 79 fotos reales de `dataset_real`.

**Limitación conocida:** 1 de las 40 fotos falsas (una réplica de Smirnoff)
coincidía a propósito con una resolución válida (1600x900) y pasó R0b. Un
atacante que además imite nitidez/brillo normales evade este chequeo -- no
es una defensa completa. Estas 40 fotos **no se agregaron** a
`dataset_real` ni se usaron para entrenar nada.

## Pendientes conocidos

- **Anti-fraude de captura:** R0b solo bloquea por resolución; falta una
  segunda capa de contenido (OCR sobre el texto de la etiqueta + detección
  de texto ilegible/gibberish) para el caso de un atacante que imite la
  resolución. La solución de fondo, para el despliegue real, es que el
  sistema capture directo desde la cámara de la Raspberry Pi en el kiosco
  en vez de aceptar cualquier archivo subido -- así ni siquiera existe la
  superficie de ataque de "subir una imagen generada".

- `dataset_real/vidrio/` ya tiene 4 fotos reales (2026-08-12, botellas de vino), pero sigue siendo muy poco volumen -- se necesitan más fotos y de más variedad de vidrio (no solo botellas de vino oscuras) para un chequeo honesto confiable.
- Integrar dataset externo de Kaggle (`mostafaabla/garbage-classification`, clases plastic/green-glass/brown-glass/white-glass) para variedad de color de vidrio -- pendiente de descarga manual (requiere login de Kaggle).
- Fine-tuning dedicado con dataset real ampliado, usando la estrategia correcta (fase final separada, no mezclada con Kaggle) una vez haya suficiente volumen de fotos propias.
- v4 confirmó que el problema no es falta de fotos reales de plástico en train (75 fotos, 480 copias oversampleadas) sino que el modelo asocia el fondo/iluminación de la caja de luz con "vidrio" -- probablemente hace falta re-pensar el preprocesamiento (recorte al objeto, normalización de brillo) en vez de solo agregar más fotos.
