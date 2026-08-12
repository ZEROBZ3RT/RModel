"""
Corre el modelo entrenado sobre TODAS las imagenes en dataset_real/plastico
(incluye las 13 originales usadas/held-out en el reentrenamiento v3 y las
nuevas agregadas despues) e imprime la prediccion + confianza de cada una.
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = r"C:\Users\luanb\Desktop\RModel"
REAL_DIR = os.path.join(ROOT, "dataset_real", "plastico")
MODEL_PATH = os.path.join(ROOT, "modelo", "model", "ecosort_mobilenetv2_v3_real.keras")
IMG_SIZE = (224, 224)

HELD_OUT = {"plastico_003.jpeg", "plastico_008.jpeg", "plastico_013.jpeg"}
ORIGINAL_13 = {f"plastico_{i:03d}.jpeg" for i in range(1, 14)}

model = tf.keras.models.load_model(MODEL_PATH)

files = sorted(os.listdir(REAL_DIR))
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

    if f in HELD_OUT:
        grupo = "held_out_v3 (nunca vista en entrenamiento)"
    elif f in ORIGINAL_13:
        grupo = "vista_en_entrenamiento_v3 (oversampleada)"
    else:
        grupo = "nueva (no usada en ningun entrenamiento)"

    results.append({
        "archivo": f, "grupo": grupo, "prediccion": pred,
        "confianza_pct": round(conf * 100, 1), "correcto": correcto,
    })
    print(f"{f:28s} [{grupo:45s}] -> {pred:9s} ({conf*100:5.1f}%) {'OK' if correcto else 'MAL'}")

n = len(results)
n_ok = sum(r["correcto"] for r in results)
print(f"\nTotal: {n} imagenes | Correctas (plastico): {n_ok} | Incorrectas: {n - n_ok} | Accuracy: {n_ok/n*100:.1f}%")

nuevas = [r for r in results if r["grupo"].startswith("nueva")]
if nuevas:
    n_ok_new = sum(r["correcto"] for r in nuevas)
    print(f"Solo imagenes NUEVAS agregadas: {len(nuevas)} | Correctas: {n_ok_new} | Accuracy: {n_ok_new/len(nuevas)*100:.1f}%")

out_path = os.path.join(ROOT, "modelo", "pred_real_plastico.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)
print(f"\nResultados guardados en {out_path}")
