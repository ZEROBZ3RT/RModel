import os, json, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFile
import imagehash
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
FIGS = ROOT / "analysis" / "figs"
SAMPLES = ROOT / "analysis" / "samples"
CLASSES = ["plastico", "vidrio"]

report = {"classes": {}}

# ---------- 1. Inventario + integridad ----------
records = []  # (class, path, ok, reason)
for cls in CLASSES:
    folder = DATASET / cls
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    for p in files:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                w, h = im.size
                mode = im.mode
            records.append({"class": cls, "path": str(p.relative_to(ROOT)), "ok": True,
                             "width": w, "height": h, "mode": mode, "size_kb": p.stat().st_size / 1024})
        except Exception as e:
            records.append({"class": cls, "path": str(p.relative_to(ROOT)), "ok": False, "reason": str(e)})

n_total = len(records)
n_corrupt = sum(1 for r in records if not r["ok"])
print(f"Total imagenes encontradas: {n_total}")
print(f"Imagenes corruptas: {n_corrupt}")

valid = [r for r in records if r["ok"]]

# ---------- 2. Distribucion de clases ----------
class_counts = defaultdict(int)
for r in valid:
    class_counts[r["class"]] += 1
report["classes"]["counts_valid"] = dict(class_counts)
report["classes"]["counts_raw"] = {cls: sum(1 for r in records if r["class"] == cls) for cls in CLASSES}

plt.figure(figsize=(5, 4))
names = list(class_counts.keys())
vals = [class_counts[n] for n in names]
bars = plt.bar(names, vals, color=["#4C9AFF", "#7A869A"])
plt.title("Distribucion de clases (imagenes validas)")
plt.ylabel("Numero de imagenes")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width()/2, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig(FIGS / "01_distribucion_clases.png", dpi=150)
plt.close()

total_valid = sum(class_counts.values())
ratio = max(class_counts.values()) / min(class_counts.values())
report["classes"]["ratio_desbalance"] = round(ratio, 3)
report["classes"]["porcentaje"] = {k: round(v/total_valid*100, 2) for k, v in class_counts.items()}

# ---------- 3. Resoluciones ----------
res = defaultdict(int)
widths, heights, sizes_kb = [], [], []
for r in valid:
    res[(r["width"], r["height"])] += 1
    widths.append(r["width"]); heights.append(r["height"]); sizes_kb.append(r["size_kb"])

report["resolutions"] = {
    "unique_resolutions": len(res),
    "most_common": sorted(res.items(), key=lambda x: -x[1])[:5],
    "width_min": min(widths), "width_max": max(widths), "width_mean": round(float(np.mean(widths)), 1),
    "height_min": min(heights), "height_max": max(heights), "height_mean": round(float(np.mean(heights)), 1),
    "filesize_kb_mean": round(float(np.mean(sizes_kb)), 1),
    "filesize_kb_min": round(float(np.min(sizes_kb)), 1),
    "filesize_kb_max": round(float(np.max(sizes_kb)), 1),
}

# ---------- 4. Duplicados exactos (MD5) y casi-duplicados (phash) ----------
md5_map = defaultdict(list)
phash_map = defaultdict(list)
for r in valid:
    fp = ROOT / r["path"]
    with open(fp, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    md5_map[md5].append(r["path"])
    try:
        with Image.open(fp) as im:
            ph = imagehash.phash(im)
        phash_map[str(ph)].append(r["path"])
    except Exception:
        pass

exact_dupes = {k: v for k, v in md5_map.items() if len(v) > 1}
near_dupes = {k: v for k, v in phash_map.items() if len(v) > 1}
n_exact_dupe_files = sum(len(v) - 1 for v in exact_dupes.values())
n_near_dupe_files = sum(len(v) - 1 for v in near_dupes.values())

report["duplicates"] = {
    "exact_duplicate_groups": len(exact_dupes),
    "exact_duplicate_extra_files": n_exact_dupe_files,
    "near_duplicate_groups_phash": len(near_dupes),
    "near_duplicate_extra_files_phash": n_near_dupe_files,
    "example_exact_groups": list(exact_dupes.values())[:3],
}

# ---------- 5. Calidad visual: blur (Laplacian variance) y brillo ----------
blur_scores = {"plastico": [], "vidrio": []}
brightness = {"plastico": [], "vidrio": []}
for r in valid:
    fp = ROOT / r["path"]
    img = cv2.imread(str(fp))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_scores[r["class"]].append(lap_var)
    brightness[r["class"]].append(float(np.mean(gray)))

BLUR_THRESHOLD = 100.0  # por debajo de esto se considera borrosa (heuristica comun)
DARK_THRESHOLD = 60.0
BRIGHT_THRESHOLD = 200.0

quality = {}
for cls in CLASSES:
    b = np.array(blur_scores[cls])
    br = np.array(brightness[cls])
    quality[cls] = {
        "n": len(b),
        "blur_var_mean": round(float(b.mean()), 1),
        "blur_var_min": round(float(b.min()), 1),
        "pct_borrosas_lt100": round(float((b < BLUR_THRESHOLD).mean() * 100), 2),
        "brightness_mean": round(float(br.mean()), 1),
        "pct_oscuras_lt60": round(float((br < DARK_THRESHOLD).mean() * 100), 2),
        "pct_sobreexpuestas_gt200": round(float((br > BRIGHT_THRESHOLD).mean() * 100), 2),
    }
report["quality"] = quality

plt.figure(figsize=(6, 4))
plt.hist(blur_scores["plastico"], bins=40, alpha=0.6, label="plastico")
plt.hist(blur_scores["vidrio"], bins=40, alpha=0.6, label="vidrio")
plt.axvline(BLUR_THRESHOLD, color="red", linestyle="--", label="umbral borrosidad")
plt.xlim(0, 1500)
plt.xlabel("Varianza del Laplaciano (nitidez)")
plt.ylabel("Numero de imagenes")
plt.title("Distribucion de nitidez por clase")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "02_nitidez_por_clase.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.hist(brightness["plastico"], bins=40, alpha=0.6, label="plastico")
plt.hist(brightness["vidrio"], bins=40, alpha=0.6, label="vidrio")
plt.xlabel("Brillo medio (0-255)")
plt.ylabel("Numero de imagenes")
plt.title("Distribucion de brillo por clase")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "03_brillo_por_clase.png", dpi=150)
plt.close()

# ---------- 6. Muestras representativas (grid) ----------
def make_grid(cls, n=8):
    folder = DATASET / cls
    files = sorted([p for p in folder.iterdir() if p.is_file()])[:n]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, fp in zip(axes.flat, files):
        img = Image.open(fp).convert("RGB")
        ax.imshow(img)
        ax.set_title(fp.name, fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Muestras representativas: {cls}")
    plt.tight_layout()
    plt.savefig(SAMPLES / f"muestras_{cls}.png", dpi=150)
    plt.close()

for cls in CLASSES:
    make_grid(cls)

# ---------- 7. Casos de baja calidad (mas borrosos) para evidenciar hallazgos ----------
def worst_examples(cls, k=6):
    fp_list = [(ROOT / r["path"], v) for r, v in zip(
        [r for r in valid if r["class"] == cls], blur_scores[cls])]
    fp_list.sort(key=lambda x: x[1])
    return fp_list[:k]

for cls in CLASSES:
    worst = worst_examples(cls)
    fig, axes = plt.subplots(1, len(worst), figsize=(3*len(worst), 3.2))
    if len(worst) == 1:
        axes = [axes]
    for ax, (fp, score) in zip(axes, worst):
        img = Image.open(fp).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{fp.name}\nnitidez={score:.0f}", fontsize=8)
        ax.axis("off")
    plt.suptitle(f"Imagenes de menor calidad (mas borrosas): {cls}")
    plt.tight_layout()
    plt.savefig(SAMPLES / f"peor_calidad_{cls}.png", dpi=150)
    plt.close()

# ---------- Guardar reporte JSON ----------
report["totals"] = {"n_total_raw": n_total, "n_corrupt": n_corrupt, "n_valid": total_valid}
with open(ROOT / "analysis" / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(json.dumps(report, indent=2, ensure_ascii=False))
