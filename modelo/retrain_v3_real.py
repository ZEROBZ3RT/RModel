"""
Reentrenamiento v3: integra 10 fotos reales de plastico (dataset_real/plastico)
al set de entrenamiento, oversampleadas, usando los MISMOS hiperparametros
originales documentados en los informes (NO los de v2, que empeoraron el
comportamiento en fotos reales). 3 fotos reales se dejan fuera del entrenamiento
como chequeo honesto de generalizacion.
No sobrescribe el modelo original ni el v2.
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

ROOT = r"C:\Users\luanb\Desktop\RModel"
DATA_DIR = os.path.join(ROOT, "dataset_split")
REAL_DIR = os.path.join(ROOT, "dataset_real", "plastico")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OVERSAMPLE = 8

HELD_OUT = {"plastico_003.jpeg", "plastico_008.jpeg", "plastico_013.jpeg"}

# ---------- 1. Insertar fotos reales (oversampleadas) en train/plastico ----------
train_plastico_dir = os.path.join(DATA_DIR, "train", "plastico")
# limpiar copias reales de una corrida anterior, si existieran
for f in os.listdir(train_plastico_dir):
    if f.startswith("real_"):
        os.remove(os.path.join(train_plastico_dir, f))

real_files = sorted(os.listdir(REAL_DIR))
used_for_training = [f for f in real_files if f not in HELD_OUT]
print("Fotos reales usadas en entrenamiento:", used_for_training)
print("Fotos reales dejadas fuera (chequeo honesto):", sorted(HELD_OUT))

for f in used_for_training:
    src = os.path.join(REAL_DIR, f)
    base, ext = os.path.splitext(f)
    for r in range(OVERSAMPLE):
        dst = os.path.join(train_plastico_dir, f"real_{base}_rep{r}{ext}")
        shutil.copy2(src, dst)

n_real_added = len(used_for_training) * OVERSAMPLE
print(f"Copias reales agregadas a train/plastico: {n_real_added}")

# ---------- 2. Pipeline igual al original ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=SEED)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)
test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"), image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False)

class_names = train_ds.class_names
idx_plastico = class_names.index("plastico")
idx_vidrio = class_names.index("vidrio")
print("class_names:", class_names)

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


def build_model():
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

counts = {}
for split in ["train", "val", "test"]:
    counts[split] = {}
    for cls in class_names:
        d = os.path.join(DATA_DIR, split, cls)
        counts[split][cls] = len(os.listdir(d))
print(json.dumps(counts, indent=2))

n_plastico = counts["train"]["plastico"]
n_vidrio = counts["train"]["vidrio"]
total_train = n_plastico + n_vidrio
class_weight = {
    idx_plastico: total_train / (2 * n_plastico),
    idx_vidrio: total_train / (2 * n_vidrio),
}
print("class_weight:", class_weight)

# ---------- Fase 1 ----------
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])
callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]
t0 = time.time()
history_1 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=12,
                       class_weight=class_weight, callbacks=callbacks_p1, verbose=2)
t_fase1 = time.time() - t0
print(f"Tiempo fase 1: {t_fase1:.1f} s")

# ---------- Fase 2: fine-tuning ultimas 30 capas (igual al original) ----------
base_model.trainable = True
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
history_2 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=10,
                       class_weight=class_weight, callbacks=callbacks_p2, verbose=2)
t_fase2 = time.time() - t0
print(f"Tiempo fase 2: {t_fase2:.1f} s")
tiempo_total = t_fase1 + t_fase2
print(f"Tiempo total v3: {tiempo_total:.1f} s ({tiempo_total/60:.1f} min)")

# ---------- Evaluacion sobre test set oficial (403 img, sin fotos reales) ----------
y_true, y_prob = [], []
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

print("=== RESULTADOS V3 (test set oficial, 403 img.) ===")
print(f"accuracy_test = {acc_test:.4f}")
print("cm", cm.tolist())
print("weighted", prec_w, rec_w, f1_w)

# ---------- Chequeo honesto: fotos reales (10 vistas en train + 3 held-out) ----------
def eval_real(files, label):
    print(f"--- {label} ---")
    results = []
    for f in files:
        path = os.path.join(REAL_DIR, f)
        img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        prob_vidrio = float(model.predict(arr, verbose=0)[0][0])
        pred = "vidrio" if prob_vidrio >= 0.5 else "plastico"
        conf = prob_vidrio if pred == "vidrio" else 1 - prob_vidrio
        correcto = pred == "plastico"
        print(f"  {f}: pred={pred} conf={conf*100:.1f}% {'OK' if correcto else 'MAL'}")
        results.append({"archivo": f, "prediccion": pred, "confianza": round(conf*100, 1), "correcto": correcto})
    return results

res_train_reales = eval_real(used_for_training, "Fotos reales VISTAS en entrenamiento (no es prueba justa)")
res_held_out = eval_real(sorted(HELD_OUT), "Fotos reales NUNCA vistas (chequeo honesto)")

# ---------- Guardar modelo v3 ----------
model_dir = os.path.join(ROOT, "analysis", "model")
keras_path = os.path.join(model_dir, "ecosort_mobilenetv2_v3_real.keras")
model.save(keras_path)

summary = {
    "cambios_v3": [
        f"{len(used_for_training)} fotos reales de plastico integradas al train set, oversampleadas x{OVERSAMPLE} ({n_real_added} copias)",
        "3 fotos reales dejadas fuera del entrenamiento como chequeo honesto",
        "Hiperparametros IDENTICOS al modelo original (30 capas fine-tuning, sin color jitter extra)",
        "0 fotos reales de vidrio disponibles todavia -- limitacion conocida",
    ],
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
        "matriz_confusion": cm.tolist(),
    },
    "fotos_reales_vistas_en_train": res_train_reales,
    "fotos_reales_held_out": res_held_out,
}
with open(os.path.join(ROOT, "analysis", "train_report_v3_real.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("OK - modelo v3 guardado en", keras_path)
