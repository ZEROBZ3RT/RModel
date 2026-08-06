const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, ImageRun, PageBreak, Header, Footer,
  PageNumber, VerticalAlign,
} = require("docx");

const path = require("path");
const IMG = (p) => fs.readFileSync(path.join(__dirname, p));

function img(path, w, h) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ data: IMG(path), transformation: { width: w, height: h }, type: "png" })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 }, children: [new TextRun(text)] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, children: [new TextRun(text)] });
}
function p(text, opts = {}) {
  return new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text, ...opts })] });
}
function bullet(text) {
  return new Paragraph({ bullet: { level: 0 }, spacing: { after: 80 }, children: [new TextRun(text)] });
}
function strongLine(label, value) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [new TextRun({ text: label + ": ", bold: true }), new TextRun(value)],
  });
}

const CELL_SHADE = { type: ShadingType.CLEAR, fill: "EAF1FB" };

function simpleTable(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mkCell = (text, bold, shade) => new TableCell({
    width: { size: widths[0], type: WidthType.DXA },
    shading: shade ? CELL_SHADE : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text: String(text), bold })] })],
  });
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: CELL_SHADE,
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: htext, bold: true })] })],
    })),
  });
  const bodyRows = rows.map((r) => new TableRow({
    children: r.map((cellText, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 80, bottom: 80, left: 100, right: 100 },
      children: [new Paragraph({ children: [new TextRun({ text: String(cellText) })] })],
    })),
  }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [headerRow, ...bodyRows] });
}

const CONTENT_W = 9360; // Letter, 1in margins

const doc = new Document({
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "EcoSort AI — Análisis del Dataset", size: 16, color: "888888" })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Página ", size: 16 }), new TextRun({ children: [PageNumber.CURRENT], size: 16 })],
        })],
      }),
    },
    children: [
      // ---------------- PORTADA ----------------
      new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "PONTIFICIA UNIVERSIDAD CATÓLICA DEL ECUADOR", bold: true, size: 24 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "SEDE MANABÍ", bold: true, size: 24 })] }),
      new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Análisis del Dataset — Proyecto Integrador", size: 22, color: "555555" })] }),
      new Paragraph({ spacing: { before: 400 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "EcoSort AI", bold: true, size: 40 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Sistema Inteligente para la Identificación de Residuos de Plástico y Vidrio mediante Raspberry Pi e Inteligencia Artificial", size: 22, italics: true })] }),
      new Paragraph({ spacing: { before: 800 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Materias integradas: Sistemas Expertos · Tecnologías de Plataforma · Análisis y Circuitos · Manejo y Desarrollo de Proyectos", size: 18, color: "555555" })] }),
      new Paragraph({ spacing: { before: 1600 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Quinto Semestre — Ingeniería en Software", size: 20 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Julio, 2026", size: 20 })] }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---------------- 1. Descripción general del problema ----------------
      h1("1. Descripción general del problema"),
      p("El proyecto EcoSort AI busca automatizar la identificación de residuos de plástico y vidrio mediante una cámara USB conectada a una Raspberry Pi 5. Un modelo de inteligencia artificial procesa la imagen del residuo, predice su clase (plástico, vidrio o desconocido) y un módulo de sistema experto valida ese resultado aplicando reglas sobre el porcentaje de confianza antes de almacenarlo y mostrarlo en un dashboard web."),
      p("Para poder entrenar el clasificador de imágenes se requiere un dataset etiquetado de residuos de plástico y vidrio. Este informe documenta el origen, la calidad, la limpieza y la preparación del dataset elegido, siguiendo la rúbrica de análisis de dataset del proyecto integrador."),
      p("Un requisito particular del proyecto es que el modelo debe distinguir correctamente botellas de plástico transparente (PET) de botellas de vidrio, ya que ambos materiales pueden lucir visualmente similares (transparencia, brillo, forma cilíndrica) frente a una cámara. Por esto se buscó explícitamente un dataset que incluyera botellas plásticas transparentes junto a envases de vidrio."),

      // ---------------- 2. Origen del dataset ----------------
      h1("2. Origen del dataset"),
      strongLine("Nombre", "Drinking Waste Classification"),
      strongLine("Fuente", "Kaggle (kaggle.com/datasets/arkadiyhacks/drinking-waste-classification)"),
      strongLine("Autor", "Arkadiy Serezhkin"),
      strongLine("Licencia", "CC0: Public Domain (uso libre, incluido uso académico/comercial)"),
      strongLine("Contexto de recolección", "Proyecto de fin de carrera (Individual Project) en University College London (UCL). Las fotografías fueron tomadas manualmente con una cámara de teléfono de 12 MP como parte de un sistema de detección de residuos en tiempo real basado en YOLO."),
      strongLine("Procedencia mixta", "El propio autor indica que la clase PET incorpora parte del dataset TrashNet (Gary Thung y Mindy Yang, UCL/Stanford), lo cual explica los duplicados detectados en la sección 5."),
      p("El dataset original contiene 4 clases de residuos de bebidas: latas de aluminio (AluCan), botellas de vidrio (Glass), botellas PET de plástico transparente (PET) y botellas de leche HDPE (HDPEM), con un total de 14 500 archivos (1.63 GB), incluyendo además una copia anotada en formato YOLO no utilizada en este proyecto."),
      p("Pertinencia: para EcoSort AI solo interesa distinguir plástico vs. vidrio, por lo que se seleccionaron únicamente las carpetas Glass y PET, descartando AluCan (metal) y HDPEM (plástico opaco, fuera del alcance). La clase PET es además la más relevante para el caso de botellas transparentes que el sistema experto debe resolver con las reglas de confianza."),

      // ---------------- 3. Características del dataset ----------------
      h1("3. Características del dataset"),
      h2("3.1 Número total de registros"),
      simpleTable(
        ["Clase", "Archivos originales", "Descripción"],
        [
          ["plastico (PET)", "1508", "Botellas de plástico transparente (PET), fotografiadas individualmente"],
          ["vidrio (Glass)", "1232", "Botellas de vidrio (mayormente ámbar/verde), fotografiadas individualmente"],
          ["Total subset usado", "2740", "Suma de las 2 clases relevantes para EcoSort AI"],
        ],
        [3200, 2560, 3600]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      h2("3.2 Tipo de datos"),
      bullet("Datos no estructurados: imágenes en formato JPG (mayoría) y HEIC (8 archivos provenientes de iPhone)."),
      bullet("Cada imagen constituye un registro; la etiqueta de clase es implícita en la carpeta/prefijo del nombre de archivo (no existe un CSV de metadatos adicional)."),
      bullet("No hay anotaciones de bounding box en la versión \"rawimgs\" utilizada (sí existen en la carpeta YOLO_imgs del dataset original, descartada por no ser necesaria para clasificación simple)."),
      h2("3.3 Variables disponibles"),
      bullet("Imagen RGB del objeto (variable de entrada principal)."),
      bullet("Clase (plástico / vidrio) — variable objetivo, derivada del nombre de carpeta."),
      bullet("Metadatos técnicos derivados para este análisis: resolución (ancho x alto), tamaño de archivo en KB, formato."),
      p("Resoluciones encontradas: predominan 512×683 px (1747 imágenes) y variantes 512×384 / 384×512; existe 1 imagen atípica a resolución nativa de cámara (4032×3024 px). El tamaño de archivo promedio es de 142.8 KB (mínimo 5.5 KB, máximo ~3.6 MB)."),

      // ---------------- 4. EDA ----------------
      h1("4. Análisis exploratorio de datos (EDA)"),
      h2("4.1 Distribución de clases"),
      img("figs/01_distribucion_clases.png", 380, 304),
      caption("Figura 1. Distribución de clases del subset original (antes de limpieza): 1508 plástico vs. 1232 vidrio."),
      p("La proporción entre clases es 1508:1232, equivalente a una razón de desbalance de 1.23:1. Es un desbalance leve — no crítico — pero se cuantifica y se trata en la sección 6."),
      h2("4.2 Muestras representativas"),
      img("samples/muestras_plastico.png", 600, 300),
      caption("Figura 2. Muestras representativas de la clase plástico (PET)."),
      img("samples/muestras_vidrio.png", 600, 300),
      caption("Figura 3. Muestras representativas de la clase vidrio (Glass)."),
      p("Se observa que ambas clases fueron fotografiadas en condiciones muy controladas y repetitivas: el plástico siempre sobre un mismo mantel a cuadros y el vidrio siempre sobre el mismo piso de madera, con toma cenital (cámara apuntando hacia abajo) en casi todos los casos. Esto se retoma como riesgo en la sección 8."),
      h2("4.3 Calidad visual"),
      img("figs/02_nitidez_por_clase.png", 500, 333),
      caption("Figura 4. Distribución de nitidez (varianza del Laplaciano) por clase. Valores bajos indican imágenes borrosas."),
      simpleTable(
        ["Clase", "Nitidez media", "% imágenes borrosas (var<100)", "Brillo medio", "% sobreexpuestas"],
        [
          ["plastico", "3050.9", "3.38%", "134.0", "0.00%"],
          ["vidrio", "4138.6", "14.46%", "147.4", "0.57%"],
        ],
        [1800, 2200, 2400, 1600, 1360]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      p("La clase vidrio presenta una proporción notablemente mayor de imágenes borrosas (14.46% frente a 3.38% en plástico), principalmente por movimiento de cámara y tomas fuera de foco. La figura 5 muestra los casos más extremos detectados automáticamente por clase."),
      img("samples/peor_calidad_plastico.png", 520, 364),
      caption("Figura 5a. Las 6 imágenes más borrosas detectadas en la clase plástico (sombra de movimiento, encuadre muy cerrado)."),
      img("samples/peor_calidad_vidrio.png", 520, 364),
      caption("Figura 5b. Las 6 imágenes más borrosas detectadas en la clase vidrio (fuera de foco, primeros planos extremos, dominancia de color)."),
      bullet("Imágenes borrosas: confirmadas cuantitativamente (ver tabla e imágenes anteriores)."),
      bullet("Objetos parcialmente visibles: varias imágenes de \"peor calidad\" muestran solo la boca/tapa de la botella, sin el objeto completo."),
      bullet("Fondos complejos: el mantel a cuadros (plástico) y el patrón de madera (vidrio) son texturizados y se repiten en todo el dataset — no complejos en el sentido de \"ruido visual\", pero sí muy poco variados."),
      bullet("Iluminación: no se detectaron imágenes oscuras (0% con brillo medio < 60) ni prácticamente sobreexpuestas (≤0.57%); la iluminación interior es consistente en todo el dataset — lo cual es bueno para la clase actual, pero no representativo de la iluminación real de exteriores donde operará el sistema."),

      // ---------------- 5. Limpieza y preprocesamiento ----------------
      h1("5. Limpieza y preprocesamiento"),
      h2("5.1 Eliminación de duplicados"),
      p("Se calculó el hash MD5 exacto y el hash perceptual (pHash) de cada imagen. Se encontraron 70 grupos de duplicados exactos (140 archivos involucrados, 70 redundantes), y 71 grupos casi-idénticos por pHash. La totalidad de los duplicados se concentra dentro de la clase plástico (PET) — 0 casos cruzan clases — lo cual es consistente con que el autor combinó su propia captura con imágenes reutilizadas de TrashNet para esa clase. Se conservó 1 imagen por grupo y se eliminaron las 70 copias redundantes."),
      h2("5.2 Eliminación de imágenes corruptas / formato no soportado"),
      p("8 imágenes de la clase vidrio estaban en formato HEIC (formato nativo de fotos de iPhone), no legible por el decodificador estándar (Pillow) usado en el pipeline de entrenamiento — se reportaron inicialmente como \"corruptas\" al fallar la verificación. En lugar de descartarlas, se instaló soporte HEIF (pillow-heif) y se convirtieron las 8 imágenes a JPG, conservando así el 100% de los datos disponibles. No se encontraron imágenes verdaderamente corruptas (0 bytes, cabecera dañada, etc.) en el subset."),
      h2("5.3 Corrección de etiquetas erróneas"),
      p("Al no existir duplicados que crucen entre clases (plastico vs. vidrio) y al provenir cada carpeta de una fuente etiquetada manualmente por el autor original, no se identificaron errores evidentes de etiquetado. Como control adicional, se revisaron visualmente las muestras representativas (figuras 2 y 3) y los casos de menor calidad (figura 5), sin encontrar imágenes mal clasificadas."),
      h2("5.4 Redimensionamiento y normalización"),
      p("Las imágenes originales tienen resoluciones heterogéneas (512×683, 512×384, 384×512 y un outlier de 4032×3024). Para el entrenamiento se aplicará, dentro del pipeline de carga de datos (tf.data / ImageDataGenerator), redimensionamiento a 224×224 px — tamaño estándar de entrada de MobileNetV2, la arquitectura prevista para ejecutar el modelo en la Raspberry Pi 5 vía TensorFlow Lite — y normalización de los valores de píxel al rango [0,1] (o al rango específico que exige el preprocesamiento de MobileNetV2). Se optó por aplicar este paso en tiempo de carga (on-the-fly) en vez de generar una copia redimensionada en disco, para no duplicar el almacenamiento y poder ajustar el tamaño de entrada si la arquitectura final cambia."),

      // ---------------- 6. Balance del dataset ----------------
      h1("6. Balance del dataset"),
      p("Antes de la limpieza: 1508 plástico vs. 1232 vidrio (razón 1.23:1). Después de eliminar los 70 duplicados de la clase plástico: 1438 plástico vs. 1232 vidrio (razón 1.17:1)."),
      img("figs/01b_distribucion_clases_final.png", 380, 304),
      caption("Figura 6. Distribución de clases del dataset limpio final (2670 imágenes)."),
      p("Una razón de desbalance de 1.17:1 se considera leve (por debajo del umbral típico de preocupación de 1.5:1–2:1) y no exige técnicas agresivas de balanceo como SMOTE o undersampling severo. Aun así, para robustecer el entrenamiento se recomienda:"),
      bullet("Usar class_weight balanceado en Keras/TensorFlow al compilar el modelo, para que los errores en la clase minoritaria (vidrio) pesen ligeramente más durante el entrenamiento."),
      bullet("Aplicar data augmentation (rotación, flip horizontal, variación de brillo) de forma algo más intensiva sobre la clase vidrio, para acercar el número efectivo de ejemplos vistos por época."),

      // ---------------- 7. División para entrenamiento ----------------
      h1("7. División para entrenamiento"),
      simpleTable(
        ["Conjunto", "Porcentaje", "Plástico", "Vidrio", "Total"],
        [
          ["Entrenamiento", "70%", "1006", "862", "1868"],
          ["Validación", "15%", "215", "184", "399"],
          ["Prueba", "15%", "217", "186", "403"],
          ["Total", "100%", "1438", "1232", "2670"],
        ],
        [2200, 1600, 1800, 1800, 1960]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      p("La división se realizó de forma estratificada por clase (70/15/15), manteniendo en cada subconjunto la misma proporción plástico:vidrio que el dataset completo, y usando una semilla fija (random_state=42) para que el split sea reproducible."),
      p("Justificación: 70% para entrenamiento es suficiente dado el tamaño del dataset (1868 imágenes) para hacer transfer learning sobre una red preentrenada (MobileNetV2) sin sobreajustar de inmediato; 15% de validación permite monitorear el sobreajuste y ajustar hiperparámetros durante el entrenamiento; el 15% de prueba queda completamente aislado hasta la evaluación final, para obtener una medida honesta del desempeño antes de desplegar el modelo en la Raspberry Pi."),

      // ---------------- 8. Riesgos y limitaciones ----------------
      h1("8. Riesgos y limitaciones del dataset"),
      bullet("Cero imágenes tomadas en el contexto real del proyecto (universidad, cámara USB frontal sobre la Raspberry Pi 5): todas las fotos provienen de un entorno doméstico ajeno al de despliegue final."),
      bullet("Variabilidad de fondo muy limitada: el vidrio se fotografió siempre sobre el mismo piso de madera y el plástico siempre sobre el mismo mantel a cuadros — alto riesgo de que el modelo memorice el fondo en lugar del objeto (shortcut learning)."),
      bullet("Ángulo de captura uniforme: casi todas las fotos son tomas cenitales (de arriba hacia abajo); la cámara USB del proyecto probablemente capturará el objeto de frente o en ángulo, lo que representa un cambio de dominio (domain shift)."),
      bullet("Iluminación homogénea de interior: no hay variabilidad de luz natural/exterior ni de sombras duras, que sí podrían aparecer en el punto de uso real."),
      bullet("Objetos parcialmente ocultos o encuadres muy cerrados en un subconjunto de imágenes (ver figura 5), lo que reduce la información visual disponible en esos casos."),
      bullet("Redundancia original en la clase PET por mezcla con TrashNet (ya mitigada con la eliminación de 70 duplicados, pero reduce la diversidad real neta de esa clase)."),
      bullet("Formato HEIC no estándar en 8 imágenes (ya solucionado, pero indica que el pipeline de producción deberá aceptar o normalizar múltiples formatos de imagen)."),
      bullet("Riesgo de sobreajuste (overfitting): dado el fondo y ángulo repetitivos, el modelo puede lograr alta precisión en el conjunto de prueba (que comparte el mismo sesgo) pero comportarse mal frente a las fotos reales de la Raspberry Pi."),
      bullet("Ambigüedad plástico transparente vs. vidrio transparente: aunque se priorizó un dataset con botellas PET transparentes, no hay garantía de que cubra todos los casos límite (p. ej. vidrio muy claro/transparente vs. plástico transparente bajo la misma iluminación) que el sistema experto deberá resolver con el umbral de confianza."),

      // ---------------- 9. Estrategia para mejorar el dataset ----------------
      h1("9. Estrategia para mejorar el dataset"),
      bullet("Captura propia en la universidad: tomar fotografías reales con la misma Raspberry Pi 5 y cámara USB que se usará en producción, variando fondo (piso, mesa, césped/cancha), ángulo y hora del día, e incorporarlas como conjunto de fine-tuning o como parte del conjunto de prueba \"del mundo real\"."),
      bullet("Data augmentation en el pipeline de entrenamiter: rotación aleatoria (±20°), flip horizontal, cambios de brillo/contraste (±20%), zoom aleatorio y ligero recorte, para simular la variabilidad de fondo/iluminación que el dataset original no cubre."),
      bullet("Diversificar fondos deliberadamente en la captura propia (evitar repetir el mismo mantel/piso en todas las fotos), para romper la asociación fondo-clase que existe en el dataset base."),
      bullet("Incluir explícitamente más ejemplos de botellas de plástico transparente vacías y de vidrio transparente/claro fotografiadas bajo condiciones similares, para reducir el riesgo de confusión que debe resolver el sistema experto."),
      bullet("Complementar con un segundo dataset público (por ejemplo TrashNet o Garbage Classification, filtrando solo las clases glass/plastic) únicamente como fuente adicional de variabilidad visual, no como reemplazo del dataset principal."),
      bullet("Mantener el umbral de confianza del sistema experto (≥85% aceptar, 60–84% solicitar recaptura, <60% desconocido) como mecanismo de seguridad mientras el dataset y el modelo maduran con datos reales."),

      // ---------------- 10. Conclusiones ----------------
      h1("10. Conclusiones"),
      p("¿El dataset es suficiente para entrenar el modelo? Sí, en cantidad: 2670 imágenes limpias (1868 de entrenamiento) son un punto de partida razonable para un MVP basado en transfer learning con MobileNetV2, especialmente combinado con data augmentation. No es suficiente, en cambio, en variabilidad de contexto: fondo, ángulo e iluminación están muy poco representados en comparación con el entorno real de despliegue."),
      p("¿Qué problemas presenta? Un desbalance leve (1.17:1, manejable), 70 duplicados dentro de la clase plástico (ya eliminados), 8 imágenes en formato HEIC no estándar (ya convertidas), y sobre todo un fondo y ángulo de captura extremadamente repetitivos que no reflejan el punto de uso final del sistema (Raspberry Pi + cámara USB en la universidad)."),
      p("¿Qué acciones se realizarán para mitigarlos? Se aplicó ya la limpieza (deduplicación y conversión de formato) y la división estratificada 70/15/15 documentadas en este informe. Antes de considerar el modelo listo para producción, se planea complementar el dataset con capturas propias en el entorno real del proyecto y aplicar data augmentation agresiva orientada a variar fondo, ángulo e iluminación, validando el desempeño final sobre ese conjunto de prueba \"del mundo real\" y no únicamente sobre el test set derivado del dataset de Kaggle."),

      // ---------------- Referencias ----------------
      h1("Referencias"),
      p("Serezhkin, A. (2020). Drinking Waste Classification [Dataset]. Kaggle. https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification", { size: 20 }),
      p("Thung, G. & Yang, M. TrashNet [Dataset]. GitHub. https://github.com/garythung/trashnet", { size: 20 }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(__dirname + "/../Informe_Analisis_Dataset_EcoSortAI.docx", buf);
  console.log("OK, escrito Informe_Analisis_Dataset_EcoSortAI.docx");
});
