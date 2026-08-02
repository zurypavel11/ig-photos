#!/usr/bin/env python3
"""
Doplnění fotek do photos/pool/ - spouští se ručně, kdykoli chceš přidat
novou dávku fotek (např. jednou za měsíc).

Vezme všechny .jpg/.jpeg/.png ze zadané složky, zkontroluje kvalitu
(rozlišení, ostrost - stejná pravidla jako v publish_post.py), zmenší je
na max. 1600 px a zkopíruje do photos/pool/. Fotky, co neprojdou kontrolou,
skončí v photos/rejected/ s vysvětlením v konzoli.

Použití:
  python3 scripts/prepare_photos.py ~/Downloads/nazev-slozky-s-fotkami
"""
import sys
from pathlib import Path

import cv2
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "photos" / "pool"
REJECTED_DIR = REPO_ROOT / "photos" / "rejected"

MIN_DIMENSION = 1080
BLUR_THRESHOLD = 15.0
MAX_DIM = 1600
VALID_EXT = {".jpg", ".jpeg", ".png"}


def quality_ok(cv_img):
    h, w = cv_img.shape[:2]
    if min(h, w) < MIN_DIMENSION:
        return False, f"malé rozlišení ({w}x{h}, minimum {MIN_DIMENSION}px)"
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < BLUR_THRESHOLD:
        return False, f"rozmazaná (skóre ostrosti {blur:.1f}, minimum {BLUR_THRESHOLD})"
    return True, ""


def unique_name(stem, existing):
    base = stem.lower().replace(" ", "_")
    name = f"{base}.jpg"
    i = 2
    while name in existing:
        name = f"{base}_{i}.jpg"
        i += 1
    return name


def main():
    if len(sys.argv) < 2:
        print("Použití: python3 scripts/prepare_photos.py <cesta-ke-slozce-s-fotkami>")
        sys.exit(1)

    src = Path(sys.argv[1]).expanduser()
    if not src.exists():
        print(f"Složka {src} neexistuje.")
        sys.exit(1)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in VALID_EXT)
    if not files:
        print(f"Ve složce {src} jsem nenašel žádné .jpg/.jpeg/.png soubory.")
        sys.exit(1)

    existing = {p.name for p in POOL_DIR.glob("*.jpg")}
    added, skipped = 0, 0

    for p in files:
        cv_img = cv2.imread(str(p))
        if cv_img is None:
            print(f"Přeskočeno {p.name}: nejde načíst (poškozený soubor / nepodporovaný formát)")
            skipped += 1
            continue

        ok, reason = quality_ok(cv_img)
        if not ok:
            print(f"Přeskočeno {p.name}: {reason}")
            skipped += 1
            continue

        out_name = unique_name(p.stem, existing)
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(POOL_DIR / out_name, quality=90)
        existing.add(out_name)
        added += 1
        print(f"Přidáno: {out_name}")

    print(f"\nHotovo. Přidáno {added} fotek do photos/pool/, přeskočeno {skipped}.")


if __name__ == "__main__":
    main()
