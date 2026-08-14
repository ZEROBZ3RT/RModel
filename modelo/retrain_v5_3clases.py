"""
Reentrenamiento v5: pasa de clasificador binario (plastico/vidrio) a
clasificador plano de 3 clases (papel_carton, plastico, vidrio), agregando
una clase distractora explicita para que el sistema aprenda a rechazar
materiales que no puntuan (papel, carton) en vez de forzar una eleccion
entre plastico/vidrio cuando no es ninguno de los dos.

Fuentes de datos nuevas respecto a v4:
- dataset_real/plastico: 75 -> 471 fotos (396 nuevas de materiales.rar,
  camara final del proyecto, confirmadas por el usuario 2026-08-14).
- dataset_real/vidrio: 4 -> 188 fotos (184 nuevas, mismo origen).
- dataset/papel_carton: clase nueva, 997 imagenes (403 cardboard + 594
  paper) del dataset publico TrashNet (github.com/garythung/trashnet),
  sin fotos reales propias (no hay dataset_real/papel_carton todavia --
  limitacion documentada en el README).

Mismo esquema que v4: held-out honesto (~20% por clase, nunca visto en
train) para plastico/vidrio antes de entrenar, oversampling x8 de fotos
reales en train, 30 capas de fine-tuning. Unico cambio arquitectonico:
salida Dense(3, softmax) + sparse_categorical_crossentropy en vez de
Dense(1, sigmoid) + binary_crossentropy.
"""
import os, time, json, random, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset_split")
REAL_DIRS = {
    "plastico": os.path.join(ROOT, "dataset_real", "plastico"),
    "vidrio": os.path.join(ROOT, "dataset_real", "vidrio"),
}
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
OVERSAMPLE = 12            # subido de 8 a 12: mas peso a las fotos reales (materiales.rar),
                           # que es justamente la razon por la que se pasaron -- que el
                           # modelo aprenda a diferenciar con la camara/condiciones reales,
                           # no solo con el dataset generico de Kaggle
HELD_OUT_FRACTION = 0.20
MIN_HELD_OUT = 1
EPOCHS_FASE1 = 20
EPOCHS_FASE2 = 15

# ---------- 0. Split honesto train/held-out por clase (fotos reales) ----------
held_out = {}
used_for_training = {}
for cls, d in REAL_DIRS.items():
    files = sorted(os.listdir(d))
    shuffled = files[:]
    random.Random(SEED).shuffle(shuffled)
    n_held = max(MIN_HELD_OUT, round(len(shuffled) * HELD_OUT_FRACTION))
    held_out[cls] = sorted(shuffled[:n_held])
    used_for_training[cls] = sorted(shuffled[n_held:])
    print(f"[{cls}] total={len(files)} train={len(used_for_training[cls])} held_out={len(held_out[cls])}")

# ---------- 1. Insertar fotos reales (oversampleadas) en train/<clase> ----------
for cls in REAL_DIRS:
    train_cls_dir = os.path.join(DATA_DIR, "train", cls)
    for f in os.listdir(train_cls_dir):
        if f.startswith("real_"):
            os.remove(os.path.join(train_cls_dir, f))

n_real_added = {}
for cls, d in REAL_DIRS.items():
    train_cls_dir = os.path.join(DATA_DIR, "train", cls)
    for f in used_for_training[cls]:
        src = os.path.join(d, f)
        base, ext = os.path.splitext(f)
        for r in range(OVERSAMPLE):
            dst = os.path.join(train_cls_dir, f"real_{base}_rep{r}{ext}")
            shutil.copy2(src, dst)
    n_real_added[cls] = len(used_for_training[cls]) * OVERSAMPLE
    print(f"Copias reales agregadas a train/{cls}: {n_real_added[cls]}")

# ---------- 2. Pipeline (3 clases, labels enteros) ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=True, seed=SEED)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False)

class_names = train_ds.class_names  # alfabetico: papel_carton, plastico, vidrio
print("class_names:", class_names)
idx = {c: class_names.index(c) for c in class_names}

AUTOTUNE = tf.data.AUTOTUNE
train_ds_p = train_ds.prefetch(AUTOTUNE)
val_ds_p = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds_raw.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.055),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
], name="data_augmentation")


def build_model(n_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


model, base_model = build_model(len(class_names))

counts = {}
for split in ["train", "val", "test"]:
    counts[split] = {}
    for cls in class_names:
        d = os.path.join(DATA_DIR, split, cls)
        counts[split][cls] = len(os.listdir(d))
print(json.dumps(counts, indent=2))

total_train = sum(counts["train"].values())
n_classes = len(class_names)
class_weight = {
    idx[cls]: total_train / (n_classes * counts["train"][cls]) for cls in class_names
}
print("class_weight:", class_weight)

# ---------- Fase 1 ----------
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]
t0 = time.time()
history_1 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS_FASE1,
                       class_weight=class_weight, callbacks=callbacks_p1, verbose=2)
t_fase1 = time.time() - t0
print(f"Tiempo fase 1: {t_fase1:.1f} s")

# ---------- Fase 2: fine-tuning ultimas 30 capas ----------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
]
t0 = time.time()
history_2 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS_FASE2,
                       class_weight=class_weight, callbacks=callbacks_p2, verbose=2)
t_fase2 = time.time() - t0
print(f"Tiempo fase 2: {t_fase2:.1f} s")
tiempo_total = t_fase1 + t_fase2
print(f"Tiempo total v5: {tiempo_total:.1f} s ({tiempo_total/60:.1f} min)")

# ---------- Evaluacion sobre test set oficial (3 clases) ----------
y_true, y_pred = [], []
for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1).tolist())
    y_true.extend(labels.numpy().ravel().tolist())
y_true = np.array(y_true).astype(int)
y_pred = np.array(y_pred).astype(int)

acc_test = accuracy_score(y_true, y_pred)
label_order = [idx[c] for c in class_names]
prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=label_order)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred, labels=label_order)

print("=== RESULTADOS V5 (test set, 3 clases) ===")
print("class_names (orden matriz):", class_names)
print(f"accuracy_test = {acc_test:.4f}")
print("cm", cm.tolist())
print("precision", prec.tolist(), "recall", rec.tolist(), "f1", f1.tolist())
print("weighted", prec_w, rec_w, f1_w)


# ---------- Chequeo honesto: fotos reales held-out (plastico/vidrio) ----------
def eval_real(files, cls_dir, expected_label, label):
    print(f"--- {label} ---")
    results = []
    for f in files:
        path = os.path.join(cls_dir, f)
        img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)  # pixeles RAW 0-255: el modelo ya incluye
        # preprocess_input en su grafo (ver build_model). Aplicarlo aqui tambien lo
        # aplicaria DOS veces -- bug detectado 2026-08-14 (heredado de v2/v3/v4) que
        # corrompia las imagenes reales y producia falsos "0% accuracy en held-out"
        # / la falsa conclusion de que el modelo confundia plastico real con vidrio.
        arr = np.expand_dims(arr, axis=0)
        probs = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred = class_names[pred_idx]
        conf = float(probs[pred_idx])
        correcto = pred == expected_label
        print(f"  {f}: pred={pred} conf={conf*100:.1f}% probs={dict(zip(class_names, [round(float(p),3) for p in probs]))} {'OK' if correcto else 'MAL'}")
        results.append({"archivo": f, "prediccion": pred, "confianza": round(conf * 100, 1), "correcto": correcto})
    return results


resultados_held_out = {}
resultados_vistas_en_train = {}
for cls, d in REAL_DIRS.items():
    resultados_vistas_en_train[cls] = eval_real(
        used_for_training[cls], d, cls, f"[{cls}] VISTAS en entrenamiento (no es prueba justa)")
    resultados_held_out[cls] = eval_real(
        held_out[cls], d, cls, f"[{cls}] HELD-OUT -- nunca vistas (chequeo honesto)")

acc_held_out = {}
for cls in REAL_DIRS:
    r = resultados_held_out[cls]
    acc_held_out[cls] = round(sum(1 for x in r if x["correcto"]) / len(r), 4) if r else None
print("accuracy held-out por clase:", acc_held_out)

# ---------- Guardar modelo v5 ----------
model_dir = os.path.join(ROOT, "modelo", "model")
keras_path = os.path.join(model_dir, "ecosort_mobilenetv2_v5_3clases.keras")
model.save(keras_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(model_dir, "ecosort_mobilenetv2_v5_3clases.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

summary = {
    "cambios_v5": [
        "Cambio de arquitectura: clasificador plano de 3 clases (papel_carton, plastico, vidrio) "
        "en vez de binario plastico/vidrio -- salida Dense(3, softmax) + sparse_categorical_crossentropy",
        "papel_carton: clase distractora nueva, 997 imagenes de TrashNet (github.com/garythung/trashnet), "
        "sin fotos reales propias todavia",
        f"vidrio: dataset_real paso de 4 a {len(held_out['vidrio']) + len(used_for_training['vidrio'])} fotos reales "
        "(camara final del proyecto, confirmado por el usuario 2026-08-14)",
        f"plastico: dataset_real paso de 75 a {len(held_out['plastico']) + len(used_for_training['plastico'])} fotos reales",
        "Held-out honesto calculado con seed=42, ~20% por clase (minimo 1 foto), solo para plastico/vidrio",
        "Hiperparametros identicos a v2/v3/v4 (30 capas fine-tuning, sin color jitter extra)",
        "class_names orden alfabetico: " + str(class_names),
    ],
    "class_names": class_names,
    "held_out_files": held_out,
    "entrenamiento": {
        "epocas_fase1": len(history_1.history["accuracy"]),
        "epocas_fase2": len(history_2.history["accuracy"]),
        "tiempo_total_s": round(tiempo_total, 1),
    },
    "evaluacion_test_oficial": {
        "accuracy": round(float(acc_test), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "precision_por_clase": dict(zip(class_names, [round(float(p), 4) for p in prec])),
        "recall_por_clase": dict(zip(class_names, [round(float(r), 4) for r in rec])),
        "f1_por_clase": dict(zip(class_names, [round(float(x), 4) for x in f1])),
        "matriz_confusion": cm.tolist(),
        "matriz_confusion_orden": class_names,
    },
    "accuracy_held_out_por_clase": acc_held_out,
    "fotos_reales_vistas_en_train": resultados_vistas_en_train,
    "fotos_reales_held_out": resultados_held_out,
}
with open(os.path.join(ROOT, "modelo", "train_report_v5_3clases.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("OK - modelo v5 guardado en", keras_path)
