#!/usr/bin/env python3
"""
Automatizovaná publikace feed postu na Instagram.

Co dělá, v pořadí:
1. vybere další nepoužité téma z data/topics.json (respektuje týdenní rotaci pilířů)
2. vybere další nepoužitou fotku z photos/pool/
3. vygeneruje headline (na fotku) a caption (popisek pod post) přes Claude API
4. přidá text overlay na fotku (Pillow) - pozice a barva boxu se náhodně střídají, font zůstává stejný
5. commitne + pushne fotku do repa (Instagram potřebuje veřejnou URL)
6. publikuje přes Instagram Graph API (container -> publish)
7. označí téma jako použité a přesune fotku do photos/used/, commitne + pushne

Nutné proměnné prostředí:
  IG_ACCESS_TOKEN    - Instagram access token
  ANTHROPIC_API_KEY  - Anthropic API klíč (čte ho anthropic knihovna automaticky)

Spuštění nanečisto (nic nepublikuje na Instagram, jen vygeneruje a ukáže):
  DRY_RUN=1 python3 scripts/publish_post.py
"""
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import cv2
import requests
import anthropic
import smtplib
from email.mime.text import MIMEText

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
POOL_DIR = REPO_ROOT / "photos" / "pool"
USED_DIR = REPO_ROOT / "photos" / "used"
READY_DIR = REPO_ROOT / "photos" / "ready"
TOPICS_PATH = DATA_DIR / "topics.json"

GITHUB_USER = "zurypavel11"
GITHUB_REPO = "ig-photos"
IG_USER_ID = "17841401986984284"
IG_GRAPH_VERSION = "v21.0"

SONNET_INPUT_PRICE_PER_MTOK = 2.0
SONNET_OUTPUT_PRICE_PER_MTOK = 10.0


def estimate_cost_usd(input_tokens, output_tokens):
    return (input_tokens / 1_000_000 * SONNET_INPUT_PRICE_PER_MTOK) + \
           (output_tokens / 1_000_000 * SONNET_OUTPUT_PRICE_PER_MTOK)

# Kandidáti na monospace font - script vezme první, co na systému existuje.
# macOS má Courier New/Menlo předinstalované, Linux (GitHub Actions runner) DejaVu/Liberation.
FONT_PATH_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/Library/Fonts/Courier New Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]

POSITIONS = ["bottom-left", "bottom-right"]
BOX_COLORS = [(255, 255, 255, 255), (245, 245, 240, 255)]


def load_json(path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_font(size):
    for p in FONT_PATH_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    print("VAROVÁNÍ: nenašel jsem žádný z očekávaných fontů, používám default.", file=sys.stderr)
    return ImageFont.load_default()


def next_topic(topics):
    unused = [t for t in topics["topics"] if not t["used"]]
    if not unused:
        raise SystemExit("Došla témata v data/topics.json - je potřeba doplnit nová.")
    week = datetime.date.today().isocalendar()[1]
    target_pillar = (week % 5) + 1
    same_pillar = [t for t in unused if t["pillar"] == target_pillar]
    return same_pillar[0] if same_pillar else unused[0]


MIN_DIMENSION = 1080          # Instagram doporučené minimum
BLUR_THRESHOLD = 15.0          # nižší = rozmazanější (Laplacian variance)
REJECTED_DIR = REPO_ROOT / "photos" / "rejected"


def photo_quality_ok(path):
    """Vrátí (True, '') pokud je fotka dost ostrá a dost velká, jinak (False, důvod)."""
    img = cv2.imread(str(path))
    if img is None:
        return False, "nejde načíst (poškozený soubor?)"
    h, w = img.shape[:2]
    if min(h, w) < MIN_DIMENSION:
        return False, f"malé rozlišení ({w}x{h}, minimum {MIN_DIMENSION}px)"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < BLUR_THRESHOLD:
        return False, f"rozmazaná (skóre ostrosti {blur:.1f}, minimum {BLUR_THRESHOLD})"
    return True, ""


def next_photo():
    photos = sorted(POOL_DIR.glob("*.jpg")) + sorted(POOL_DIR.glob("*.jpeg"))
    if not photos:
        raise SystemExit("Ve photos/pool/ nejsou žádné fotky - je potřeba doplnit nové.")

    good = []
    REJECTED_DIR.mkdir(exist_ok=True, parents=True)
    for p in photos:
        ok, reason = photo_quality_ok(p)
        if ok:
            good.append(p)
        else:
            print(f"Přeskočena fotka {p.name}: {reason} - přesouvám do photos/rejected/")
            p.rename(REJECTED_DIR / p.name)

    if not good:
        raise SystemExit("Žádná fotka v poolu neprošla kontrolou kvality - je potřeba doplnit nové.")

    return random.choice(good)


def crop_to_feed_ratio(img, target_ratio=4 / 5):
    """Center-crop na bezpecny pomer stran pro Instagram feed (4:5), at
    Instagram sam neorizne neco jineho a text overlay nezmizi mimo
    viditelnou oblast (od ledna 2026 ma grid profilu pomer 3:4)."""
    w, h = img.size
    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img


def resize_to_max_dimension(img, max_dim=1600):
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def generate_texts(topic):
    client = anthropic.Anthropic()  # čte ANTHROPIC_API_KEY z env
    prompt = f"""Jsi Pavel Žůrek, senior PPC specialista a zakladatel marketingové agentury Tollar (výkonnostní marketing pro služby a prémiové značky, ČR). Píšeš Instagram příspěvek na téma:

Téma: {topic['title']}
Úhel/data: {topic['angle']}
Pilíř: {topic['pillar_name']}

Napiš:
1. HEADLINE - krátký úderný hook na fotku, max 6 slov, česky, bez emoji, bez uvozovek
2. CAPTION - plný popisek pod příspěvek, 100-180 slov, věcný tón, s konkrétními čísly z podkladu, zakonči otázkou pro komentáře a 3-5 hashtagy (#B2Bmarketing #Tollar apod.)

Odpověz PŘESNĚ v tomto formátu, nic navíc:
HEADLINE: <text>
CAPTION: <text>"""
    msg = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    cost = estimate_cost_usd(msg.usage.input_tokens, msg.usage.output_tokens)

    headline, caption = "", ""
    in_caption = False
    for line in text.splitlines():
        if line.startswith("HEADLINE:"):
            headline = line.replace("HEADLINE:", "").strip()
        elif line.startswith("CAPTION:"):
            caption = line.replace("CAPTION:", "").strip()
            in_caption = True
        elif in_caption:
            caption += "\n" + line
    return headline, caption.strip(), cost


def send_email_notification(subject, body):
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_address or not gmail_app_password:
        print("VAROVANI: GMAIL_ADDRESS/GMAIL_APP_PASSWORD nejsou nastavene - e-mail se neodesila.")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_app_password)
            server.send_message(msg)
        print("E-mail notifikace odeslana.")
    except Exception as e:
        print(f"VAROVANI: odeslani e-mailu selhalo: {e}")


def wrap_headline(headline):
    words = headline.split()
    if len(words) <= 3:
        return [headline]
    mid = len(words) // 2 + len(words) % 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def add_overlay(photo_path, headline, out_path):
    img = Image.open(photo_path).convert("RGBA")
    img = crop_to_feed_ratio(img, target_ratio=4 / 5)
    img = resize_to_max_dimension(img, max_dim=1350)
    w, h = img.size
    font_size = int(w * 0.052)
    font = pick_font(font_size)
    pad_x, pad_y = int(w * 0.05), int(h * 0.025)
    line_spacing = int(font_size * 1.25)

    lines = wrap_headline(headline)

    draw_tmp = ImageDraw.Draw(img)
    line_widths = [draw_tmp.textbbox((0, 0), l, font=font)[2] for l in lines]
    box_w = max(line_widths) + pad_x * 2
    box_h = line_spacing * len(lines) + pad_y * 2

    position = random.choice(POSITIONS)
    margin = int(w * 0.06)
    box_y = h - box_h - margin if "bottom" in position else (
        margin if "top" in position else (h - box_h) // 2
    )
    box_x = margin if "left" in position else w - box_w - margin

    color = random.choice(BOX_COLORS)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=int(font_size * 0.35), fill=color,
    )
    ty = box_y + pad_y
    for line in lines:
        odraw.text((box_x + pad_x, ty), line, font=font, fill=(15, 15, 15, 255))
        ty += line_spacing

    combined = Image.alpha_composite(img, overlay)
    combined.convert("RGB").save(out_path, quality=90)


def git(*args):
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def publish_to_instagram(image_url, caption):
    token = os.environ["IG_ACCESS_TOKEN"]
    base = f"https://graph.instagram.com/{IG_GRAPH_VERSION}/{IG_USER_ID}"
    r = requests.post(f"{base}/media", data={
        "image_url": image_url, "caption": caption, "access_token": token,
    })
    r.raise_for_status()
    creation_id = r.json()["id"]
    time.sleep(5)
    r2 = requests.post(f"{base}/media_publish", data={
        "creation_id": creation_id, "access_token": token,
    })
    r2.raise_for_status()
    return r2.json()["id"]


def main():
    topics = load_json(TOPICS_PATH, {"topics": []})
    topic = next_topic(topics)
    photo = next_photo()

    print(f"Téma [{topic['pillar_name']}]: {topic['title']}")
    print(f"Fotka: {photo.name}")

    headline, caption, ai_cost = generate_texts(topic)
    print(f"\nHeadline: {headline}\n\nCaption:\n{caption}\n")
    print(f"Odhadovana cena AI generovani: ${ai_cost:.4f}")

    READY_DIR.mkdir(exist_ok=True, parents=True)
    out_name = f"post_{topic['id']}_{photo.stem}.jpg"
    out_path = READY_DIR / out_name
    add_overlay(photo, headline, out_path)
    print(f"Fotka s overlayem: {out_path}")

    rel_path = out_path.relative_to(REPO_ROOT)
    git("add", str(rel_path))
    git("commit", "-m", f"post: {topic['title'][:60]}")
    git("push")

    image_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{rel_path.as_posix()}"
    print(f"Veřejná URL: {image_url}")

    if os.environ.get("DRY_RUN") == "1":
        print("\nDRY_RUN=1 - přeskočena skutečná publikace na Instagram.")
        return

    post_id = publish_to_instagram(image_url, caption)
    print(f"\nPublikováno! Post ID: {post_id}")

    send_email_notification(
        subject=f"IG post publikovan: {topic['title'][:60]}",
        body=(
            f"Novy Instagram post byl publikovan na @zurypavel.\n\n"
            f"Tema [{topic['pillar_name']}]: {topic['title']}\n\n"
            f"Headline na fotce: {headline}\n\n"
            f"Popisek:\n{caption}\n\n"
            f"Fotka: {photo.name}\n"
            f"Odkaz na fotku: {image_url}\n"
            f"Instagram post ID: {post_id}\n\n"
            f"Odhadovana cena AI generovani textu: ${ai_cost:.4f}\n"
        ),
    )

    topic["used"] = True
    topic["used_date"] = datetime.date.today().isoformat()
    topic["post_id"] = post_id
    save_json(TOPICS_PATH, topics)

    new_photo_path = USED_DIR / photo.name
    git_targets = [str(TOPICS_PATH.relative_to(REPO_ROOT))]
    if photo.exists():
        photo.rename(new_photo_path)
        git_targets.append(str(new_photo_path.relative_to(REPO_ROOT)))
    else:
        print(f"VAROVANI: {photo} uz v photos/pool/ nebyla nalezena (mozna soubezny beh), "
              f"preskakuji presun do photos/used/ - stav tematu se presto ulozi.")

    git("add", *git_targets)
    git("commit", "-m", f"state: published topic {topic['id']} ({post_id})")
    git("push")


if __name__ == "__main__":
    main()
