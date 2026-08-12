"""
Corre el modelo entrenado sobre las imagenes nuevas recibidas por WhatsApp
(dos tandas descargadas el 2026-08-12) para verificar clasificacion plastico/vidrio.
"""
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = r"C:\Users\luanb\Desktop\RModel"
TEST_DIR = r"C:\Users\luanb\AppData\Local\Temp\claude\C--Users-luanb-Desktop-RModel\25feb097-ba5d-4c6f-8c67-37753d68c737\scratchpad\whatsapp_test"
MODEL_PATH = os.path.join(ROOT, "modelo", "model", "ecosort_mobilenetv2_v3_real.keras")
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

results = []
for subset in ["set1", "set2"]:
    subset_dir = os.path.join(TEST_DIR, subset)
    for f in sorted(os.listdir(subset_dir)):
        path = os.path.join(subset_dir, f)
        img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        prob_vidrio = float(model.predict(arr, verbose=0)[0][0])
        pred = "vidrio" if prob_vidrio >= 0.5 else "plastico"
        conf = prob_vidrio if pred == "vidrio" else 1 - prob_vidrio

        results.append({
            "tanda": subset, "archivo": f,
            "prediccion": pred, "confianza_pct": round(conf * 100, 1),
        })
        print(f"[{subset}] {f:45s} -> {pred:9s} ({conf*100:5.1f}%)")

n = len(results)
n_plastico = sum(1 for r in results if r["prediccion"] == "plastico")
print(f"\nTotal: {n} imagenes | Predichas plastico: {n_plastico} | Predichas vidrio: {n - n_plastico}")

out_path = os.path.join(ROOT, "modelo", "pred_whatsapp_test.json")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2, ensure_ascii=False)
print(f"\nResultados guardados en {out_path}")
