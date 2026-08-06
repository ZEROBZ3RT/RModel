"""Genera el notebook Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb (celdas, sin ejecutar).
Se ejecuta despues con: jupyter nbconvert --to notebook --execute --inplace <notebook>
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""# EcoSort AI — Implementación y Evaluación del Modelo de IA
Clasificador binario de residuos **plástico vs. vidrio** mediante transfer learning con **MobileNetV2**,
siguiendo el diseño aprobado en el informe *Diseño de la Solución de IA* (Sistemas Expertos, PUCE Manabí, 2026).

Dataset: `dataset_split/` (train/val/test, split estratificado 70/15/15 ya generado y documentado
en el informe de análisis del dataset).""")

code("""import os, time, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              accuracy_score, ConfusionMatrixDisplay)
import pandas as pd

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "../dataset_split"
OUT_DIR = "model"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("figs_impl", exist_ok=True)

tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPUs disponibles:", tf.config.list_physical_devices("GPU"))""")

md("## 1. Carga de datos\nSe reutiliza la partición ya generada (70/15/15) — no se vuelve a mezclar aleatoriamente.")

code("""train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=SEED)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)

class_names = train_ds.class_names
test_file_paths = test_ds_raw.file_paths
print("Clases (orden usado por el modelo):", class_names)
print("Batches -> train:", tf.data.experimental.cardinality(train_ds).numpy(),
      "val:", tf.data.experimental.cardinality(val_ds).numpy(),
      "test:", tf.data.experimental.cardinality(test_ds_raw).numpy())""")

code("""# Conteo real de imagenes por conjunto/clase (para class_weight y para el informe)
counts = {}
for split in ["train", "val", "test"]:
    counts[split] = {}
    for cls in class_names:
        d = os.path.join(DATA_DIR, split, cls)
        counts[split][cls] = len([f for f in os.listdir(d)])
print(json.dumps(counts, indent=2))""")

md("""## 2. Preparación del pipeline (augmentation + preprocesamiento)
Data augmentation aplicado solo en entrenamiento: rotación, flip horizontal, zoom y variación de contraste,
para compensar la baja variabilidad de fondo/ángulo detectada en el análisis del dataset.""")

code("""data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.055),   # ~20 grados
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds_raw.prefetch(AUTOTUNE)""")

md("""## 3. Definición del modelo (MobileNetV2, transfer learning)
Arquitectura: entrada 224x224x3 -> augmentation -> `preprocess_input` (rango [-1,1] esperado por MobileNetV2)
-> base MobileNetV2 preentrenada en ImageNet (sin la cabeza de clasificación) -> GlobalAveragePooling ->
Dropout(0.2) -> Dense(1, sigmoid).""")

code("""def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

model, base_model = build_model()
model.summary()""")

md("""## 4. Entrenamiento — Fase 1 (base congelada)
Se entrena únicamente la cabeza de clasificación. Optimizador Adam (lr=1e-4), pérdida binary crossentropy,
`class_weight` balanceado por el desbalance leve (train: plástico > vidrio), `EarlyStopping` y
`ReduceLROnPlateau` sobre `val_loss`.""")

code("""n_plastico = counts["train"][class_names[0]] if class_names[0] == "plastico" else counts["train"]["plastico"]
n_vidrio = counts["train"]["vidrio"]
total_train = n_plastico + n_vidrio
# class_names[0] es la clase mapeada a la etiqueta 0 por image_dataset_from_directory (orden alfabetico)
idx_plastico = class_names.index("plastico")
idx_vidrio = class_names.index("vidrio")
class_weight = {
    idx_plastico: total_train / (2 * n_plastico),
    idx_vidrio: total_train / (2 * n_vidrio),
}
print("class_weight:", class_weight)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

t0 = time.time()
history_1 = model.fit(train_ds, validation_data=val_ds, epochs=12,
                       class_weight=class_weight, callbacks=callbacks_p1, verbose=2)
t_fase1 = time.time() - t0
print(f"Tiempo fase 1: {t_fase1:.1f} s")""")

md("""## 5. Entrenamiento — Fase 2 (fine-tuning)
Se descongelan las últimas ~30 capas de la base MobileNetV2 y se continúa el entrenamiento con una
tasa de aprendizaje mucho menor (1e-5), para especializar los filtros preentrenados sin destruirlos.""")

code("""base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
]

t0 = time.time()
history_2 = model.fit(train_ds, validation_data=val_ds, epochs=10,
                       class_weight=class_weight, callbacks=callbacks_p2, verbose=2)
t_fase2 = time.time() - t0
print(f"Tiempo fase 2: {t_fase2:.1f} s")
tiempo_entrenamiento_total = t_fase1 + t_fase2
print(f"Tiempo total de entrenamiento: {tiempo_entrenamiento_total:.1f} s "
      f"({tiempo_entrenamiento_total/60:.1f} min)")""")

md("## 6. Curvas de entrenamiento (Accuracy y Loss vs. Épocas)")

code("""acc = history_1.history["accuracy"] + history_2.history["accuracy"]
val_acc = history_1.history["val_accuracy"] + history_2.history["val_accuracy"]
loss = history_1.history["loss"] + history_2.history["loss"]
val_loss = history_1.history["val_loss"] + history_2.history["val_loss"]
epochs_range = range(1, len(acc) + 1)
fine_tune_epoch = len(history_1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs_range, acc, label="Entrenamiento")
axes[0].plot(epochs_range, val_acc, label="Validación")
axes[0].axvline(fine_tune_epoch + 0.5, color="gray", linestyle="--", linewidth=1, label="Inicio fine-tuning")
axes[0].set_title("Accuracy vs. Épocas"); axes[0].set_xlabel("Época"); axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(epochs_range, loss, label="Entrenamiento")
axes[1].plot(epochs_range, val_loss, label="Validación")
axes[1].axvline(fine_tune_epoch + 0.5, color="gray", linestyle="--", linewidth=1, label="Inicio fine-tuning")
axes[1].set_title("Loss vs. Épocas"); axes[1].set_xlabel("Época"); axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("figs_impl/01_curvas_entrenamiento.png", dpi=160, bbox_inches="tight")
plt.show()

print(f"Accuracy final train: {acc[-1]:.4f} | val: {val_acc[-1]:.4f}")
print(f"Loss final train: {loss[-1]:.4f} | val: {val_loss[-1]:.4f}")""")

md("""## 7. Evaluación del modelo (conjunto de prueba)
Métricas sobre el 15% de prueba, aislado durante todo el entrenamiento: accuracy, precision, recall,
F1-score (por clase y ponderado), matriz de confusión y tiempo promedio de inferencia por imagen.""")

code("""y_true, y_prob = [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0).ravel()
    y_prob.extend(probs.tolist())
    y_true.extend(labels.numpy().ravel().tolist())

y_true = np.array(y_true).astype(int)
y_prob = np.array(y_prob)
y_pred = (y_prob >= 0.5).astype(int)

acc_test = accuracy_score(y_true, y_pred)
prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[idx_plastico, idx_vidrio])
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred, labels=[idx_plastico, idx_vidrio])

print(f"Accuracy (test): {acc_test:.4f}")
for i, cls in enumerate(["plastico", "vidrio"]):
    print(f"  {cls:9s} -> precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}  n={support[i]}")
print(f"  weighted -> precision={prec_w:.3f}  recall={rec_w:.3f}  f1={f1_w:.3f}")
print("Matriz de confusion (filas=real, columnas=prediccion) [plastico, vidrio]:")
print(cm)""")

code("""disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["plastico", "vidrio"])
fig, ax = plt.subplots(figsize=(5, 4.5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Matriz de confusión — conjunto de prueba")
plt.tight_layout()
plt.savefig("figs_impl/02_matriz_confusion.png", dpi=160, bbox_inches="tight")
plt.show()""")

code("""# Tiempo promedio de inferencia por imagen (CPU), sobre una imagen individual
single_batch = next(iter(test_ds.unbatch().batch(1)))
imgs_single, _ = single_batch
_ = model.predict(imgs_single, verbose=0)  # warm-up

n_reps = 30
t0 = time.time()
for _ in range(n_reps):
    _ = model.predict(imgs_single, verbose=0)
t_inferencia_ms = (time.time() - t0) / n_reps * 1000
print(f"Tiempo promedio de inferencia por imagen: {t_inferencia_ms:.1f} ms (CPU, batch=1, promedio de {n_reps} corridas)")""")

md("""## 8. Pruebas con imágenes nuevas del conjunto de prueba
Tabla de casos individuales (imagen, clase real, predicción, confianza, correcta) — mínimo 20 pruebas,
tomadas del conjunto de prueba que el modelo no vio durante el entrenamiento.""")

code("""rng = np.random.default_rng(SEED)
n_samples = 24
sample_idx = rng.choice(len(test_file_paths), size=min(n_samples, len(test_file_paths)), replace=False)

rows = []
for i in sample_idx:
    fname = os.path.basename(test_file_paths[i])
    real = class_names[y_true[i]]
    pred = class_names[y_pred[i]]
    conf = y_prob[i] if y_pred[i] == 1 else 1 - y_prob[i]
    rows.append({
        "archivo": fname, "clase_real": real, "prediccion": pred,
        "confianza": round(float(conf) * 100, 1), "correcta": real == pred,
    })

df_pruebas = pd.DataFrame(rows)
df_pruebas.to_csv("pruebas_modelo.csv", index=False)
print(f"{len(df_pruebas)} pruebas generadas -> aciertos: {df_pruebas['correcta'].sum()} / {len(df_pruebas)}")
df_pruebas""")

code("""from PIL import Image

def contact_sheet(indices, filename, ncols=6, title=None):
    n = len(indices)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.1, nrows * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax_i, idx in enumerate(indices):
        img = Image.open(test_file_paths[idx]).convert("RGB").resize((160, 160))
        ax = axes[ax_i]
        ax.imshow(img)
        real = class_names[y_true[idx]]
        pred = class_names[y_pred[idx]]
        ok = real == pred
        color = "#2E7D32" if ok else "#C62828"
        ax.set_title(f"real:{real}\\npred:{pred}", fontsize=8, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.5)
        ax.set_xticks([]); ax.set_yticks([])
    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=160, bbox_inches="tight")
    plt.show()

contact_sheet(sample_idx[:18], "figs_impl/03_muestras_pruebas.png",
              title="Muestra de pruebas sobre el conjunto de test")""")

md("""## 9. Análisis de errores
¿Dónde falla el modelo? Se identifican los casos mal clasificados con mayor confianza (los "errores más seguros"),
que suelen indicar ambigüedad real entre plástico transparente y vidrio transparente.""")

code("""wrong_mask = y_pred != y_true
wrong_idx = np.where(wrong_mask)[0]
print(f"Errores totales en test: {len(wrong_idx)} / {len(y_true)} ({len(wrong_idx)/len(y_true)*100:.1f}%)")

err_by_class = {}
for cls_idx, cls_name in enumerate(["plastico", "vidrio"]):
    real_cls = np.where(y_true == (idx_plastico if cls_name=='plastico' else idx_vidrio))[0]
    wrong_cls = np.intersect1d(real_cls, wrong_idx)
    err_by_class[cls_name] = {"total": len(real_cls), "errores": len(wrong_cls)}
print(json.dumps(err_by_class, indent=2))

if len(wrong_idx) > 0:
    wrong_conf = np.where(y_pred[wrong_idx] == 1, y_prob[wrong_idx], 1 - y_prob[wrong_idx])
    order = wrong_idx[np.argsort(-wrong_conf)]
    peores = order[:min(12, len(order))]
    contact_sheet(peores, "figs_impl/04_peores_errores.png",
                  title="Errores con mayor confianza (casos más ambiguos)")
else:
    print("No se registraron errores en el conjunto de prueba.")""")

md("""**Preguntas de análisis:**
- *¿Dónde falla el modelo?* Ver figura de errores anterior — se concentran principalmente en imágenes con
  transparencia ambigua o encuadres/fondos atípicos frente al resto del dataset.
- *¿Qué imágenes generan confusión?* Las de mayor confianza equivocada (mostradas arriba) — indican qué tan
  "seguro" estaba el modelo al equivocarse, no solo que se equivocó.
- *¿Por qué ocurre?* Consistente con los riesgos anticipados en el informe de diseño: fondo/ángulo poco
  variados en el dataset de origen y ambigüedad visual plástico transparente vs. vidrio transparente.""")

md("## 10. Exportación del modelo entrenado")

code("""model_path_keras = os.path.join(OUT_DIR, "ecosort_mobilenetv2.keras")
model.save(model_path_keras)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
model_path_tflite = os.path.join(OUT_DIR, "ecosort_mobilenetv2.tflite")
with open(model_path_tflite, "wb") as f:
    f.write(tflite_model)

size_keras_mb = os.path.getsize(model_path_keras) / (1024 * 1024)
size_tflite_mb = os.path.getsize(model_path_tflite) / (1024 * 1024)
print(f"Modelo .keras: {size_keras_mb:.2f} MB")
print(f"Modelo .tflite (cuantizado): {size_tflite_mb:.2f} MB")""")

md("## 11. Resumen de métricas (para el informe)")

code("""summary = {
    "clases": class_names,
    "conteos": counts,
    "entrenamiento": {
        "epocas_fase1": len(history_1.history["accuracy"]),
        "epocas_fase2": len(history_2.history["accuracy"]),
        "tiempo_fase1_s": round(t_fase1, 1),
        "tiempo_fase2_s": round(t_fase2, 1),
        "tiempo_total_s": round(tiempo_entrenamiento_total, 1),
        "accuracy_train_final": round(float(acc[-1]), 4),
        "accuracy_val_final": round(float(val_acc[-1]), 4),
        "loss_train_final": round(float(loss[-1]), 4),
        "loss_val_final": round(float(val_loss[-1]), 4),
    },
    "evaluacion_test": {
        "accuracy": round(float(acc_test), 4),
        "precision_plastico": round(float(prec[0]), 4),
        "recall_plastico": round(float(rec[0]), 4),
        "f1_plastico": round(float(f1[0]), 4),
        "precision_vidrio": round(float(prec[1]), 4),
        "recall_vidrio": round(float(rec[1]), 4),
        "f1_vidrio": round(float(f1[1]), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "matriz_confusion": cm.tolist(),
        "n_test": int(len(y_true)),
        "tiempo_inferencia_ms": round(float(t_inferencia_ms), 1),
    },
    "errores": err_by_class,
    "modelo": {
        "tamano_keras_mb": round(size_keras_mb, 2),
        "tamano_tflite_mb": round(size_tflite_mb, 2),
    },
    "pruebas_muestra": rows,
}

with open("train_report.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Resumen guardado en analysis/train_report.json")
print(json.dumps(summary["evaluacion_test"], indent=2))""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (venv313)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}

with open("Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook escrito: Implementacion_Evaluacion_Modelo_EcoSortAI.ipynb")
