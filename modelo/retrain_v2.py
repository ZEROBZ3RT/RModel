"""
Reentrenamiento v2 de EcoSort AI: agrega aumento de color (hue/saturacion) para
reducir el atajo "transparente=plastico / con tinte de color=vidrio" detectado
al probar el modelo original con una botella PET de tinte azul-verdoso.
No sobrescribe el modelo original (ecosort_mobilenetv2.keras), guarda uno nuevo
(ecosort_mobilenetv2_v2.keras) para no invalidar los informes ya entregados.
"""
import os, time, json, random
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
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

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


class ColorJitter(layers.Layer):
    """Jitter de hue/saturacion, solo activo en entrenamiento (igual que las demas
    capas de preprocesamiento de Keras: no-op en inferencia/evaluacion)."""
    def __init__(self, hue_delta=0.08, sat_lower=0.6, sat_upper=1.4, **kwargs):
        super().__init__(**kwargs)
        self.hue_delta = hue_delta
        self.sat_lower = sat_lower
        self.sat_upper = sat_upper

    def call(self, x, training=None):
        if training:
            x = x / 255.0
            x = tf.image.random_hue(x, self.hue_delta)
            x = tf.image.random_saturation(x, self.sat_lower, self.sat_upper)
            x = tf.clip_by_value(x, 0.0, 1.0) * 255.0
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hue_delta": self.hue_delta, "sat_lower": self.sat_lower, "sat_upper": self.sat_upper})
        return cfg


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.055),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    ColorJitter(hue_delta=0.08, sat_lower=0.5, sat_upper=1.5),
], name="data_augmentation_v2")


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


model, base_model = build_model()
model.summary()

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

# ---------- Fase 1: base congelada ----------
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

# ---------- Fase 2: fine-tuning, ultimas 50 capas (antes 30) ----------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])
callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
]
t0 = time.time()
history_2 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=14,
                       class_weight=class_weight, callbacks=callbacks_p2, verbose=2)
t_fase2 = time.time() - t0
print(f"Tiempo fase 2: {t_fase2:.1f} s")
tiempo_total = t_fase1 + t_fase2
print(f"Tiempo total entrenamiento v2: {tiempo_total:.1f} s ({tiempo_total/60:.1f} min)")

# ---------- Evaluacion sobre el mismo test set (403 img) ----------
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

print("=== RESULTADOS V2 (mismo test set de 403 img.) ===")
print(f"accuracy_test = {acc_test:.4f}")
for i, cls in enumerate(["plastico", "vidrio"]):
    print(f"  {cls:9s} -> precision={prec[i]:.4f} recall={rec[i]:.4f} f1={f1[i]:.4f} n={support[i]}")
print(f"  weighted -> precision={prec_w:.4f} recall={rec_w:.4f} f1={f1_w:.4f}")
print("Matriz de confusion [plastico, vidrio] (filas=real, columnas=prediccion):")
print(cm)

# ---------- Guardar modelo v2 (NO sobrescribe el original) ----------
model_dir = os.path.join(ROOT, "analysis", "model")
keras_path = os.path.join(model_dir, "ecosort_mobilenetv2_v2.keras")
model.save(keras_path)
size_keras_mb = os.path.getsize(keras_path) / (1024 * 1024)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_path = os.path.join(model_dir, "ecosort_mobilenetv2_v2.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
size_tflite_mb = os.path.getsize(tflite_path) / (1024 * 1024)

summary = {
    "clases": class_names,
    "conteos": counts,
    "cambios_v2": [
        "ColorJitter (hue +/-0.08, saturacion 0.5-1.5x) agregado a data_augmentation",
        "Fine-tuning en las ultimas 50 capas de MobileNetV2 (antes 30)",
        "Dropout 0.3 (antes 0.2)",
        "Fase 2: hasta 14 epocas (antes 10), patience=5 (antes 4)",
    ],
    "entrenamiento": {
        "epocas_fase1": len(history_1.history["accuracy"]),
        "epocas_fase2": len(history_2.history["accuracy"]),
        "tiempo_fase1_s": round(t_fase1, 1),
        "tiempo_fase2_s": round(t_fase2, 1),
        "tiempo_total_s": round(tiempo_total, 1),
        "accuracy_train_final": round(float(history_2.history["accuracy"][-1]), 4),
        "accuracy_val_final": round(float(history_2.history["val_accuracy"][-1]), 4),
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
    },
    "modelo": {
        "tamano_keras_mb": round(size_keras_mb, 2),
        "tamano_tflite_mb": round(size_tflite_mb, 2),
    },
}
with open(os.path.join(ROOT, "analysis", "train_report_v2.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("OK - modelo v2 guardado en", keras_path)
