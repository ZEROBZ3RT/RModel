const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, ImageRun, PageBreak, Header, Footer,
  PageNumber, VerticalAlign,
} = require("docx");

const R = JSON.parse(fs.readFileSync(path.join(__dirname, "train_report.json"), "utf-8"));

const IMG = (p) => fs.readFileSync(path.join(__dirname, p));
function img(imgPath, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ data: IMG(imgPath), transformation: { width: w, height: h }, type: "png" })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
  });
}
function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 }, children: [new TextRun(text)] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun(text)] }); }
function p(text, opts = {}) { return new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text, ...opts })] }); }
function bullet(text) { return new Paragraph({ bullet: { level: 0 }, spacing: { after: 80 }, children: [new TextRun(text)] }); }
function strongLine(label, value) {
  return new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: label + ": ", bold: true }), new TextRun(String(value))] });
}

const CELL_SHADE = { type: ShadingType.CLEAR, fill: "EAF1FB" };
function simpleTable(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA }, shading: CELL_SHADE, verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: htext, bold: true })] })],
    })),
  });
  const bodyRows = rows.map((r) => new TableRow({
    children: r.map((cellText, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: String(cellText) })] })],
    })),
  }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}

// ---------- Derivados de train_report.json ----------
const ent = R.entrenamiento;
const ev = R.evaluacion_test;
const mdl = R.modelo;
const errs = R.errores;
const pruebas = R.pruebas_muestra;
const nEpocasTotal = ent.epocas_fase1 + ent.epocas_fase2;
const minutosTotal = (ent.tiempo_total_s / 60).toFixed(1);

const pruebasRows = pruebas.map((row) => [
  row.archivo, row.clase_real, row.prediccion, `${row.confianza}%`, row.correcta ? "Sí" : "No",
]);
const aciertosPruebas = pruebas.filter((r) => r.correcta).length;

const cm = ev.matriz_confusion; // [[plastico->plastico, plastico->vidrio],[vidrio->plastico, vidrio->vidrio]]

function veredictoFinal() {
  if (ev.accuracy >= 0.9) return "Sí, cumple ampliamente el objetivo planteado en el diseño (accuracy de prueba ≥90%).";
  if (ev.accuracy >= 0.8) return "Cumple de forma aceptable el objetivo, aunque con margen de mejora antes de un despliegue productivo.";
  return "Cumple parcialmente el objetivo; se requieren ajustes adicionales (más datos reales, más fine-tuning) antes de considerarlo listo para producción.";
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "EcoSort AI — Implementación y Evaluación del Modelo de IA", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Página ", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })] }) },
    children: [
      // ---------------- PORTADA ----------------
      new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "PONTIFICIA UNIVERSIDAD CATÓLICA DEL ECUADOR", bold: true, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "SEDE MANABÍ", bold: true, size: 24 })] }),
      new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Implementación y Evaluación del Modelo de IA — Proyecto Integrador", size: 22, color: "555555" })] }),
      new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "EcoSort AI", bold: true, size: 40 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Sistema Inteligente para la Identificación de Residuos de Plástico y Vidrio mediante Raspberry Pi e Inteligencia Artificial", size: 22, italics: true })] }),
      new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Materia: Sistemas Expertos — Examen Final", size: 18, color: "555555" })] }),
      new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Quinto Semestre — Ingeniería en Software", size: 20 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Julio, 2026", size: 20 })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- 1. Introducción ----------------
      h1("1. Introducción"),
      h2("1.1 Objetivo de la implementación"),
      p("Este informe documenta la implementación, entrenamiento y evaluación del modelo de inteligencia artificial de EcoSort AI, encargado de clasificar residuos de plástico y vidrio a partir de una imagen. El objetivo de esta fase es materializar y validar experimentalmente la solución descrita en el informe de diseño previo, midiendo su desempeño real sobre datos que el modelo no vio durante el entrenamiento."),
      h2("1.2 Relación con el diseño previamente aprobado"),
      p("La implementación sigue punto por punto el diseño planteado en el informe \"Diseño de la Solución de IA\": arquitectura MobileNetV2 preentrenada en ImageNet con transfer learning en dos fases (base congelada y luego fine-tuning), clasificación binaria plástico/vidrio, y la misma partición de datos 70/15/15 documentada en el análisis del dataset. No se realizaron cambios de arquitectura respecto a lo diseñado; esta fase se limita a implementar, entrenar y medir esa propuesta."),

      // ---------------- 2. Tecnologías empleadas ----------------
      h1("2. Tecnologías empleadas"),
      simpleTable(
        ["Herramienta", "Uso en el proyecto"],
        [
          ["Python 3.13", "Lenguaje principal de todo el pipeline de datos, entrenamiento y evaluación."],
          ["TensorFlow / Keras", "Definición, entrenamiento y exportación del modelo MobileNetV2."],
          ["TensorFlow Lite", "Conversión del modelo entrenado a formato .tflite, optimizado para inferencia en la Raspberry Pi 5."],
          ["scikit-learn", "Cálculo de métricas de evaluación: precision, recall, F1-score y matriz de confusión."],
          ["Matplotlib / Pillow", "Generación de las gráficas de entrenamiento, matriz de confusión y contactos de imágenes de prueba."],
          ["Jupyter Notebook", "Documentación reproducible y ejecutable de todo el proceso (Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb)."],
          ["Node.js + docx", "Generación automatizada de este informe en formato Word/PDF, reutilizando las métricas calculadas en el notebook."],
        ],
        [3200, 6160]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),

      // ---------------- 3. Preparación de los datos ----------------
      h1("3. Preparación de los datos"),
      p("Se reutilizó el dataset ya limpio y dividido, documentado en el informe de análisis del dataset (deduplicación de 70 imágenes redundantes en la clase plástico, conversión de 8 imágenes HEIC a JPG, sin imágenes corruptas remanentes)."),
      simpleTable(
        ["Conjunto", "Plástico", "Vidrio", "Total"],
        [
          ["Entrenamiento", R.conteos.train.plastico, R.conteos.train.vidrio, R.conteos.train.plastico + R.conteos.train.vidrio],
          ["Validación", R.conteos.val.plastico, R.conteos.val.vidrio, R.conteos.val.plastico + R.conteos.val.vidrio],
          ["Prueba", R.conteos.test.plastico, R.conteos.test.vidrio, R.conteos.test.plastico + R.conteos.test.vidrio],
        ],
        [2400, 2400, 2400, 2160]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      bullet("Redimensionamiento: todas las imágenes se llevan a 224×224 px, la resolución de entrada esperada por MobileNetV2."),
      bullet("Normalización: se aplica preprocess_input de MobileNetV2 (reescala los píxeles al rango [-1, 1])."),
      bullet("Data augmentation (solo en entrenamiento): flip horizontal, rotación aleatoria (±20°), zoom aleatorio (±15%) y variación de contraste (±20%), para compensar la baja variabilidad de fondo/ángulo detectada en el análisis exploratorio."),
      bullet("División Train/Validation/Test: se conservó la partición estratificada 70/15/15 con semilla fija (random_state=42), sin volver a mezclar los conjuntos en esta fase."),

      // ---------------- 4. Implementación del modelo ----------------
      h1("4. Implementación del modelo"),
      strongLine("Arquitectura implementada", "MobileNetV2 (ImageNet) + GlobalAveragePooling2D + Dropout(0.2) + Dense(1, sigmoid)"),
      strongLine("Número de capas", "Base MobileNetV2 (~154 capas) + 3 capas propias añadidas sobre la base"),
      strongLine("Función de activación de salida", "Sigmoide (clasificación binaria)"),
      strongLine("Función de pérdida", "Binary Crossentropy"),
      strongLine("Optimizador", "Adam"),
      strongLine("Learning Rate", "1e-4 en fase 1 (base congelada); 1e-5 en fase 2 (fine-tuning), con ReduceLROnPlateau"),
      strongLine("Batch Size", "32"),
      strongLine("Número de épocas", `${nEpocasTotal} efectivas (fase 1: ${ent.epocas_fase1}, fase 2: ${ent.epocas_fase2}; con early stopping sobre val_loss)`),
      h2("4.1 Transfer Learning"),
      strongLine("Modelo base", "MobileNetV2, preentrenado en ImageNet"),
      strongLine("Capas congeladas", "Fase 1: toda la base MobileNetV2. Fase 2: todas menos las últimas 30 capas."),
      strongLine("Capas entrenadas", "Fase 1: solo el clasificador superior (GlobalAveragePooling + Dropout + Dense). Fase 2: las últimas 30 capas de la base + clasificador superior."),

      // ---------------- 5. Entrenamiento ----------------
      h1("5. Entrenamiento"),
      p(`El entrenamiento se ejecutó en CPU (sin GPU dedicada), en dos fases, con una duración total de ${ent.tiempo_total_s} s (~${minutosTotal} min): ${ent.tiempo_fase1_s} s para la fase 1 (base congelada) y ${ent.tiempo_fase2_s} s para la fase 2 (fine-tuning).`),
      simpleTable(
        ["Métrica", "Entrenamiento", "Validación"],
        [
          ["Accuracy (última época)", ent.accuracy_train_final, ent.accuracy_val_final],
          ["Loss (última época)", ent.loss_train_final, ent.loss_val_final],
        ],
        [3200, 3200, 3160]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      img("figs_impl/01_curvas_entrenamiento.png", 560, 210),
      caption("Figura 1. Accuracy y Loss (entrenamiento vs. validación) por época. La línea punteada marca el inicio del fine-tuning (fase 2)."),

      // ---------------- 6. Evaluación del modelo ----------------
      h1("6. Evaluación del modelo"),
      p(`Métricas calculadas sobre el conjunto de prueba (${ev.n_test} imágenes), aislado durante todo el entrenamiento:`),
      simpleTable(
        ["Métrica", "Plástico", "Vidrio", "Ponderado"],
        [
          ["Precision", ev.precision_plastico, ev.precision_vidrio, ev.precision_weighted],
          ["Recall", ev.recall_plastico, ev.recall_vidrio, ev.recall_weighted],
          ["F1-score", ev.f1_plastico, ev.f1_vidrio, ev.f1_weighted],
        ],
        [2400, 2200, 2200, 2760]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      strongLine("Accuracy global (test)", ev.accuracy),
      strongLine("Tiempo promedio de inferencia por imagen (CPU)", `${ev.tiempo_inferencia_ms} ms`),
      img("figs_impl/02_matriz_confusion.png", 340, 306),
      caption("Figura 2. Matriz de confusión sobre el conjunto de prueba."),
      p(`Lectura de la matriz: de ${cm[0][0] + cm[0][1]} imágenes reales de plástico, ${cm[0][0]} se clasificaron correctamente y ${cm[0][1]} se confundieron con vidrio; de ${cm[1][0] + cm[1][1]} imágenes reales de vidrio, ${cm[1][1]} se clasificaron correctamente y ${cm[1][0]} se confundieron con plástico.`),

      // ---------------- 7. Pruebas del modelo ----------------
      h1("7. Pruebas del modelo"),
      p(`Se evaluaron ${pruebas.length} imágenes nuevas tomadas del conjunto de prueba (no vistas en entrenamiento ni validación), con ${aciertosPruebas} aciertos de ${pruebas.length} (${((aciertosPruebas/pruebas.length)*100).toFixed(1)}%). El listado completo también se adjunta como pruebas_modelo.csv.`),
      img("figs_impl/03_muestras_pruebas.png", 560, 420),
      caption("Figura 3. Muestra visual de casos de prueba (borde verde = acierto, borde rojo = error)."),
      simpleTable(
        ["Archivo", "Clase real", "Predicción", "Confianza", "Correcta"],
        pruebasRows,
        [2800, 2000, 2000, 1600, 960]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),

      // ---------------- 8. Análisis de errores ----------------
      h1("8. Análisis de errores"),
      h2("¿Dónde falla el modelo?"),
      p(`De ${errs.plastico.total} imágenes reales de plástico en el conjunto de prueba, ${errs.plastico.errores} se clasificaron incorrectamente; de ${errs.vidrio.total} imágenes reales de vidrio, ${errs.vidrio.errores} se clasificaron incorrectamente. La figura 4 muestra los errores en los que el modelo tenía mayor confianza (los casos más "seguros" en los que se equivocó), que son los más informativos para entender la causa raíz.`),
      img("figs_impl/04_peores_errores.png", 480, 320),
      caption("Figura 4. Errores de clasificación con mayor confianza del modelo (casos más ambiguos)."),
      h2("¿Qué imágenes generan confusión?"),
      p("Principalmente imágenes con transparencia ambigua (plástico PET transparente vs. vidrio claro) y encuadres o fondos que se apartan del patrón dominante del dataset de origen (mantel a cuadros para plástico, piso de madera para vidrio)."),
      h2("¿Por qué ocurre?"),
      p("Es consistente con los riesgos anticipados en el informe de diseño: el dataset de origen tiene fondo y ángulo de captura muy poco variados, por lo que el modelo pudo aprender asociaciones parciales entre fondo y clase (shortcut learning) en vez de basarse exclusivamente en las propiedades visuales del material; además, la ambigüedad óptica real entre plástico y vidrio transparentes es un límite físico del problema, no solo un defecto del modelo."),

      // ---------------- 9. Integración con el prototipo ----------------
      h1("9. Integración con el prototipo"),
      p("El modelo se exporta en dos formatos: un archivo .keras (para continuar el desarrollo/depuración en PC) y un archivo .tflite cuantizado, pensado para ejecutarse con el intérprete de TensorFlow Lite directamente en la Raspberry Pi 5. La integración a nivel de software sigue el contrato definido en la arquitectura del informe de diseño: el módulo de captura entrega una imagen 224×224 ya normalizada al intérprete TFLite, este devuelve la probabilidad de la clase \"vidrio\", y esa salida se entrega tal cual al módulo de sistema experto (reglas de confianza) para la decisión final y su registro en el dashboard. El ensamblaje físico de la cámara, la Raspberry Pi y el dashboard corresponde a las materias de Tecnologías de Plataforma y Manejo y Desarrollo de Proyectos dentro de este mismo proyecto integrado, por lo que aquí se documenta únicamente el contrato de integración del modelo, no el montaje del hardware."),

      // ---------------- 10. Conclusiones ----------------
      h1("10. Conclusiones"),
      h2("¿Cumple el modelo el objetivo?"),
      p(veredictoFinal()),
      h2("¿Qué precisión alcanzó?"),
      p(`El modelo alcanzó un accuracy de ${(ev.accuracy * 100).toFixed(1)}% sobre el conjunto de prueba, con un F1-score ponderado de ${ev.f1_weighted} (plástico: ${ev.f1_plastico}, vidrio: ${ev.f1_vidrio}), y un tiempo de inferencia promedio de ${ev.tiempo_inferencia_ms} ms por imagen en CPU, compatible con clasificación en tiempo real en la Raspberry Pi 5.`),
      h2("¿Qué mejoraría?"),
      bullet("Incorporar imágenes reales capturadas en la universidad con la misma cámara/Raspberry Pi del proyecto, para reducir el riesgo de domain shift identificado en el análisis de errores."),
      bullet("Añadir ejemplos adicionales de materiales transparentes ambiguos (plástico PET transparente vs. vidrio claro), el caso de error más frecuente detectado."),
      bullet("Probar un segundo ciclo de fine-tuning con más capas descongeladas o más épocas si el presupuesto de tiempo lo permite, monitoreando de cerca el sobreajuste."),
      bullet("Validar el modelo .tflite exportado directamente sobre hardware Raspberry Pi real, para confirmar que el tiempo de inferencia medido en PC se mantiene dentro de un rango aceptable en el dispositivo final."),

      // ---------------- Entregables ----------------
      h1("Entregables de esta actividad"),
      bullet("Informe (este documento): Informe_Implementacion_Evaluacion_Modelo_EcoSortAI.docx / .pdf"),
      bullet("Código fuente y notebook ejecutado: Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb (carpeta analysis/)"),
      bullet(`Modelo entrenado: analysis/model/ecosort_mobilenetv2.keras (${mdl.tamano_keras_mb} MB) y ecosort_mobilenetv2.tflite (${mdl.tamano_tflite_mb} MB)`),
      bullet("Tabla completa de pruebas: analysis/pruebas_modelo.csv"),
      bullet("Video de 3-5 min mostrando la clasificación en funcionamiento: pendiente de grabación por el estudiante (no generado por este proceso)."),

      // ---------------- Referencias ----------------
      h1("Referencias"),
      p("Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018. https://arxiv.org/abs/1801.04381", { size: 20 }),
      p("TensorFlow. (s.f.). TensorFlow Lite — Deploy ML models on mobile and edge devices. https://www.tensorflow.org/lite", { size: 20 }),
      p("Informe interno: Diseño de la Solución de IA — EcoSort AI (PUCE Manabí, 2026), documento previo de este mismo proyecto integrado.", { size: 20 }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(__dirname + "/../Informe_Implementacion_Evaluacion_Modelo_EcoSortAI.docx", buf);
  console.log("OK, escrito Informe_Implementacion_Evaluacion_Modelo_EcoSortAI.docx");
});
