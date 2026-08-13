"""
Reentrenamiento v4: primera vez que se integran fotos REALES de vidrio
(dataset_real/vidrio, 4 fotos nuevas recibidas por WhatsApp) junto con el
dataset real de plastico ya existente (75 fotos, incluye 24 nuevas).

Antes de entrenar se separa, por clase, un held-out set (~20%, minimo 1 foto)
que NUNCA se usa en entrenamiento -- sirve de chequeo honesto post-entrenamiento
(el "test/val" de las fotos reales, pedido explicitamente antes de re-entrenar).

Hiperparametros IDENTICOS al modelo original y a v3 (mismos que documentan los
informes): 30 capas de fine-tuning, sin augmentation extra. Unico cambio real
respecto a v3: ahora SI hay fotos reales de vidrio para inyectar en train.
No sobrescribe el modelo oficial ni v2/v3.
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
BATCH_SIZE = 32
OVERSAMPLE = 8               # identico a v3
HELD_OUT_FRACTION = 0.20     # ~20% por clase, honesto, nunca visto en train
MIN_HELD_OUT = 1             # asegura al menos 1 foto held-out incluso con pocas fotos (caso vidrio)

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
    print(f"  held_out: {held_out[cls]}")

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
print(f"Tiempo total v4: {tiempo_total:.1f} s ({tiempo_total/60:.1f} min)")

# ---------- Evaluacion sobre test set oficial (403 img, Kaggle, sin fotos reales) ----------
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

print("=== RESULTADOS V4 (test set oficial Kaggle, 403 img.) ===")
print(f"accuracy_test = {acc_test:.4f}")
print("cm", cm.tolist())
print("weighted", prec_w, rec_w, f1_w)


# ---------- Chequeo honesto: fotos reales held-out (nunca vistas) + vistas en train ----------
def eval_real(files, cls_dir, expected_label, label):
    print(f"--- {label} ---")
    results = []
    for f in files:
        path = os.path.join(cls_dir, f)
        img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        prob_vidrio = float(model.predict(arr, verbose=0)[0][0])
        pred = "vidrio" if prob_vidrio >= 0.5 else "plastico"
        conf = prob_vidrio if pred == "vidrio" else 1 - prob_vidrio
        correcto = pred == expected_label
        print(f"  {f}: pred={pred} conf={conf*100:.1f}% {'OK' if correcto else 'MAL'}")
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

# ---------- Guardar modelo v4 ----------
model_dir = os.path.join(ROOT, "modelo", "model")
keras_path = os.path.join(model_dir, "ecosort_mobilenetv2_v4_vidrio.keras")
model.save(keras_path)

summary = {
    "cambios_v4": [
        "Primera vez que se integran fotos reales de VIDRIO al entrenamiento",
        f"vidrio: {len(used_for_training['vidrio'])} fotos reales usadas en train (oversampleadas x{OVERSAMPLE} = {n_real_added['vidrio']} copias), {len(held_out['vidrio'])} held-out",
        f"plastico: {len(used_for_training['plastico'])} fotos reales usadas en train (oversampleadas x{OVERSAMPLE} = {n_real_added['plastico']} copias), {len(held_out['plastico'])} held-out",
        "Held-out honesto calculado con seed=42, ~20% por clase (minimo 1 foto)",
        "Hiperparametros IDENTICOS al modelo original y a v3 (30 capas fine-tuning, sin color jitter extra)",
    ],
    "held_out_files": held_out,
    "entrenamiento": {
        "epocas_fase1": len(history_1.history["accuracy"]),
        "epocas_fase2": len(history_2.history["accuracy"]),
        "tiempo_total_s": round(tiempo_total, 1),
    },
    "evaluacion_test_oficial_kaggle": {
        "accuracy": round(float(acc_test), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "matriz_confusion": cm.tolist(),
    },
    "accuracy_held_out_por_clase": acc_held_out,
    "fotos_reales_vistas_en_train": resultados_vistas_en_train,
    "fotos_reales_held_out": resultados_held_out,
}
with open(os.path.join(ROOT, "modelo", "train_report_v4_vidrio.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("OK - modelo v4 guardado en", keras_path)
