# RePoints -- Proyecto Integrador

Sistema inteligente para identificar residuos de plástico y vidrio (y rechazar papel/cartón sin puntuar) con Raspberry Pi e IA. PUCE Manabí, Quinto Semestre, Ingeniería en Software.

## Estructura del proyecto

```
modelo/               Pipeline de datos, entrenamiento y evaluación del modelo de IA
├── Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb   Notebook principal (todo el pipeline)
├── clean_and_split.py                                  Limpieza del dataset + split 70/15/15 (3 clases)
├── analyze.py                                           EDA (duplicados, nitidez, brillo)
├── retrain_v2.py, retrain_v3_real.py, retrain_v4_vidrio.py   Experimentos anteriores (ver notas abajo)
├── retrain_v5_3clases.py                                Reentrenamiento actual: 3 clases + fotos reales ampliadas
├── model/                                                Modelos entrenados (.keras / .tflite)
├── *.json                                                Reportes de resultados (report.json, train_report*.json)
└── figs/, figs_impl/, preview*/, samples/                Figuras e imágenes generadas para los informes

sistema_experto/       Motor de reglas que valida la predicción del modelo antes de decidir y calcula puntos
└── sistema_experto.py  Clase SistemaExperto: R0b (origen/resolución) + R0 (calidad de cámara) + R1/R2/R3 (confianza) + puntaje

dataset/                Dataset base: Kaggle "Drinking Waste Classification" (plastico/vidrio)
                         + TrashNet (github.com/garythung/trashnet) para papel_carton
dataset_split/          Dataset limpio, dividido en train/val/test (generado por clean_and_split.py)
dataset_real/           Fotos reales propias (fuera del dominio de Kaggle/TrashNet), para fine-tuning
├── plastico/            431 fotos reales de plastico (incluye materiales.rar, camara final del proyecto)
├── vidrio/               171 fotos reales de vidrio (idem)
└── Plan_Captura_Dataset.xlsx   Checklist para coordinar la sesión de fotos

informes/               Entregables oficiales: los 3 informes (Análisis, Diseño, Implementación),
                         el guion de presentación y el documento de defensa oral (docx + pdf)

rubricas/                Rúbricas de evaluación del proyecto integrador
```

## Modelos entrenados

| Modelo | Descripción | Estado |
|---|---|---|
| `modelo/model/ecosort_mobilenetv2_v5_3clases.keras` / `.tflite` | **Candidato a modelo oficial (2026-08-14).** Cambia de binario (plastico/vidrio) a clasificador plano de 3 clases (papel_carton, plastico, vidrio) con `papel_carton` como clase distractora que no puntua. Ver detalle abajo. | ✅ Usar este |
| `modelo/model/ecosort_mobilenetv2.keras` / `.tflite` | Modelo binario anterior, documentado en los 3 informes. Accuracy test 97.02%. Reemplazado por v5 (agrega clase distractora + mejor volumen de fotos reales). | ⚠️ Superado por v5, dejar de referencia |
| `modelo/model/ecosort_mobilenetv2_v2.keras` / `.tflite` | Experimento: más augmentation de color + más fine-tuning. Mejoró en test set (98.51%) pero empeoró en fotos reales (más confiadamente incorrecto). | ❌ No usar |
| `modelo/model/ecosort_mobilenetv2_v3_real.keras` | Experimento: 10 fotos reales integradas al train set. No mejoró (accuracy test bajó a 96.77%, fotos reales siguieron mal clasificadas *segun el eval original -- ver bug de doble preprocesamiento abajo, este resultado no se re-verifico*). | ❌ No usar |
| `modelo/model/ecosort_mobilenetv2_v4_vidrio.keras` | Experimento: primera vez con fotos reales de vidrio (4, antes 0) + dataset real de plastico ampliado (75, antes 51). Held-out honesto (~20%, nunca visto en train) antes de entrenar. Test set oficial casi igual (97.02%). El "sesgo" reportado originalmente (15/15 fotos reales de plastico held-out prediciendose como vidrio) **era un bug de medicion, no del modelo** -- ver seccion siguiente. Re-evaluado correctamente 2026-08-14: predice esas mismas fotos como plastico con >99% de confianza. | ⚠️ Reemplazado por v5, pero su "sesgo" documentado es un bug de eval, no del modelo |

### Hallazgo critico (2026-08-14): el "sesgo caja de luz -> vidrio" de v2/v3/v4 era un bug de doble preprocesamiento, no un problema del modelo

Al re-entrenar v5 aparecio el mismo patron que v3/v4: 0% accuracy en fotos reales held-out,
todas prediciendose como la misma clase con altisima confianza (94-97%). Investigando la
causa (en vez de asumir que era el mismo sesgo de siempre) se encontro que la funcion
`eval_real()` de v2/v3/v4/v5 le aplicaba `preprocess_input()` **dos veces** a cada imagen:
una vez a mano antes de `model.predict()`, y otra vez automaticamente dentro del modelo
(que ya lo incluye en su grafo, ver `build_model()`). MobileNetV2 escala pixeles de
`[0,255]` a `[-1,1]`; aplicarlo dos veces colapsa casi todos los valores a una banda
angosta cerca de `-1`, destruyendo la imagen -- el modelo estaba clasificando practicamente
ruido, no la foto real.

Verificacion (`plastico_001.jpeg`, evaluado con el modelo v4 ya entrenado, sin reentrenar):

| | P(vidrio) segun v4 |
|---|---|
| Con el bug (doble `preprocess_input`, como en el reporte original) | 0.890 (=> predice "vidrio", **incorrecto**) |
| Corregido (una sola vez, la que hace el modelo internamente) | 0.005 (=> predice "plastico", **correcto**) |

Se corrigio `modelo/retrain_v5_3clases.py` y se re-corrio la evaluacion de fotos reales
held-out de v5 con el fix (`modelo/train_report_v5_3clases.json`, campo
`NOTA_CORRECCION_BUG_2026_08_14`). El resultado real de v5 es **95.3% en plastico
held-out y 100% en vidrio held-out** -- nada que ver con el 0%/0% que salia con el bug.

**Implicacion para v2/v3/v4:** sus conclusiones de "sesgo sistematico hacia vidrio" y
"agregar mas fotos reales no lo corrige" estan basadas en esta misma medicion rota (se
confirmo el mismo patron en v4 arriba). No se alcanzo a re-verificar v2/v3 por tiempo,
pero el mecanismo del bug es identico en los tres scripts -- **conviene tratar esa
conclusion como no confirmada** en vez de citarla en la defensa como un hallazgo solido.
Es muy probable que el modelo original tampoco tuviera el sesgo que se le atribuyo.

### v5: clasificador de 3 clases (papel_carton, plastico, vidrio)

Cambios respecto a v4:
- **Arquitectura:** salida `Dense(3, softmax)` + `sparse_categorical_crossentropy` en vez de
  `Dense(1, sigmoid)` binario. `class_names` en orden alfabetico: `papel_carton, plastico, vidrio`.
- **Clase distractora `papel_carton`:** 995 imagenes (cardboard+paper) de
  [TrashNet](https://github.com/garythung/trashnet) (github, sin necesidad de login) -- el sistema
  ahora puede reconocer y **rechazar sin puntuar** objetos que no son ni plastico ni vidrio, en vez de
  forzar una eleccion binaria. **Limitacion:** no hay fotos reales propias de papel/carton todavia
  (`dataset_real/papel_carton` no existe), asi que esta clase no tiene el mismo chequeo honesto
  contra la camara real del kiosco que si tienen plastico y vidrio.
- **dataset_real ampliado:** `plastico` 75->431 fotos, `vidrio` 4->171 fotos (lote grande recibido
  como `materiales.rar`, confirmado por el usuario como tomado con la camara final del proyecto
  dentro de la caja de luz). Se corrio limpieza de duplicados/casi-duplicados (pHash) antes de
  entrenar: 40 de plastico, 17 de vidrio y 2 de papel_carton eliminados por ser casi-identicos.
- **Hiperparametros:** batch size 32->64, oversample de fotos reales 8x->12x, epocas 12+10->20+15.

**Resultados (evaluacion corregida, ver bug arriba):**

| Metrica | Valor |
|---|---|
| Accuracy test oficial (3 clases, Kaggle+TrashNet) | 98.01% |
| Precision / Recall / F1 por clase | papel_carton 0.993/0.980/0.987 -- plastico 0.969/0.991/0.980 -- vidrio 0.984/0.968/0.976 |
| Accuracy held-out plastico (fotos reales nunca vistas) | **95.3%** (82/86) |
| Accuracy held-out vidrio (fotos reales nunca vistas) | **100%** (34/34) |

Ver `modelo/train_report_v5_3clases.json` para la matriz de confusion completa y el detalle
foto por foto.

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

Desde v5 el sistema experto trabaja con la distribucion de probabilidad de 3 clases
(`Clase.PAPEL_CARTON`, `Clase.PLASTICO`, `Clase.VIDRIO`) en vez de un unico P(vidrio)
binario -- ver `SistemaExperto.evaluar(probabilidades, tamano)`.

**Puntaje:**

| Clase | Puntos base | Con bono de botella (x2) |
|---|---|---|
| plastico | 5 | 10 |
| vidrio | 10 | 20 |
| papel_carton | 0 (no puntua, se reconoce y se rechaza) | 0 |

El bono de botella (`TamanoObjeto.BOTELLA_PEQUENA` / `BOTELLA_GRANDE`) duplica el
puntaje de plastico/vidrio; no aplica a papel_carton. Ambos valores (puntos base y
multiplicador) son parametros del constructor de `SistemaExperto`, configurables sin
tocar el modelo -- ajustar `puntos_base` / `multiplicador_botella` si el criterio de
puntaje cambia (el valor exacto del bono -- x2 vs. un monto fijo -- quedo pendiente de
confirmar con el equipo, ver Pendientes conocidos).

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
- **Re-verificar v2 y v3 con el fix del bug de doble preprocesamiento**
  (ver sección de v5 arriba) -- solo se re-confirmó v4. Si el tiempo lo
  permite antes de la defensa, vale la pena re-correr su evaluación de
  fotos reales para saber si esos experimentos tampoco tenían el problema
  que se les atribuyó.
- **`dataset_real/papel_carton` no existe todavía** -- la clase distractora
  de v5 se entrenó y evaluó 100% con TrashNet (fotos "de internet"), sin
  ningún chequeo honesto contra la cámara real del kiosco. Es el mismo
  punto ciego que tenía vidrio antes del 2026-08-12: hace falta una sesión
  de fotos propias de papel/cartón.
- **Puntaje del bono de botella sin confirmar:** se implementó como x2 (duplica
  plástico=10, vidrio=20) por decisión rápida bajo presión de tiempo -- el
  equipo mencionó que también podría ser un monto fijo (+0.5) en vez de
  multiplicador. Confirmar el criterio real antes de la defensa y ajustar
  `multiplicador_botella` en `SistemaExperto` si hace falta.
- `dataset_real/vidrio` pasó de 4 a 171 fotos reales (2026-08-14, lote
  `materiales.rar`), pero sigue siendo principalmente botellas de vino --
  conviene sumar variedad de vidrio (frascos, otros colores) cuando se
  pueda.
- Integrar dataset externo de Kaggle (`mostafaabla/garbage-classification`, clases plastic/green-glass/brown-glass/white-glass) para variedad de color de vidrio -- pendiente de descarga manual (requiere login de Kaggle).
- Fine-tuning dedicado con dataset real ampliado, usando la estrategia correcta (fase final separada, no mezclada con Kaggle) una vez haya suficiente volumen de fotos propias.
- Subclases transparente/color (plástico transparente vs. de color, vidrio
  transparente vs. de color) quedaron fuera de v5 -- las fotos de
  `materiales.rar` no vinieron separadas por color. Si se necesita esa
  granularidad, la forma más simple sigue siendo un clasificador plano con
  más clases de salida (ver discusión de diseño 2026-08-14), no un pipeline
  jerárquico de dos etapas.
