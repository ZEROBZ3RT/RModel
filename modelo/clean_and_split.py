import hashlib, json, random, shutil
from pathlib import Path
from collections import defaultdict

import pillow_heif
pillow_heif.register_heif_opener()
from PIL import Image

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
SPLIT = ROOT / "dataset_split"
CLASSES = ["plastico", "vidrio", "papel_carton"]

log = {"heic_converted": [], "duplicates_removed": [], "final_counts": {}, "split_counts": {}}

# ---------- 1. Convertir HEIC -> JPG ----------
for cls in CLASSES:
    folder = DATASET / cls
    for p in sorted(folder.glob("*.HEIC")) + sorted(folder.glob("*.heic")):
        try:
            im = Image.open(p).convert("RGB")
            im.load()
            new_path = p.with_suffix(".jpg")
            im.save(new_path, "JPEG", quality=95)
            p.unlink()
            log["heic_converted"].append(str(new_path.relative_to(ROOT)))
        except Exception as e:
            log.setdefault("heic_failed", []).append((str(p), str(e)))

# ---------- 2. Eliminar duplicados exactos (MD5), conservando 1 por grupo ----------
md5_map = defaultdict(list)
for cls in CLASSES:
    for p in sorted((DATASET / cls).iterdir()):
        if not p.is_file():
            continue
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        md5_map[md5].append(p)

for md5, paths in md5_map.items():
    if len(paths) > 1:
        for p in paths[1:]:
            log["duplicates_removed"].append(str(p.relative_to(ROOT)))
            p.unlink()

# ---------- 3. Conteo final por clase ----------
final_files = {}
for cls in CLASSES:
    files = sorted([p for p in (DATASET / cls).iterdir() if p.is_file()])
    final_files[cls] = files
    log["final_counts"][cls] = len(files)

# ---------- 4. Split estratificado 70/15/15 ----------
if SPLIT.exists():
    shutil.rmtree(SPLIT)

for split_name in ["train", "val", "test"]:
    for cls in CLASSES:
        (SPLIT / split_name / cls).mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    files = final_files[cls][:]
    random.shuffle(files)
    n = len(files)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy2(f, SPLIT / "train" / cls / f.name)
    for f in val_files:
        shutil.copy2(f, SPLIT / "val" / cls / f.name)
    for f in test_files:
        shutil.copy2(f, SPLIT / "test" / cls / f.name)

    log["split_counts"][cls] = {"train": len(train_files), "val": len(val_files), "test": len(test_files)}

with open(ROOT / "analysis" / "clean_report.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)

print(json.dumps(log, indent=2, ensure_ascii=False))
