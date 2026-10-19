"""Contact sheet of the shortlist: rank-labeled grid -> gallery/contact-sheet.png.

The Level-2 'sorting viewer' — eyeball one image, then stamp by rank number.
Usage: uv run python tools/contact_sheet.py [--prefix gallery] [--cols 4]
"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", type=Path, default=Path("gallery"))
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--thumb", type=int, default=240)
a = ap.parse_args()

rows = []
for f in sorted(a.prefix.rglob("score.json")):
    s = json.loads(f.read_text())
    best = s.get("best", {})
    if best.get("reward") is not None and best.get("reward", -1) >= 0:
        png = Path(best["png"])
        if not png.exists():  # moved since the run: the png sits by its score.json
            png = f.parent / png.name
        rows.append((best["reward"], str(png), s["prompt"][:40]))
rows.sort(reverse=True)

tw = a.thumb
th = int(tw * 1.25)
cols = a.cols
pages = [rows[i:i + 24] for i in range(0, len(rows), 24)] or [[]]
out_paths = []
for p, page in enumerate(pages, 1):
    r = (len(page) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, r * (th + 22)), "#222")
    d = ImageDraw.Draw(sheet)
    for i, (reward, png, prompt) in enumerate(page):
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        try:
            im = Image.open(png).convert("RGB").resize((tw, th))
            sheet.paste(im, (x, y + 22))
        except Exception:
            pass
        d.text((x + 4, y + 4), f"#{i+1}  r={reward:.2f}", fill="#ffd")
        d.text((x + 4, y + th + 6), prompt[:34], fill="#bbb")
    suffix = f"-{p}" if len(pages) > 1 else ""
    out = a.prefix / f"contact-sheet{suffix}.png"
    sheet.save(out)
    out_paths.append(out)
    print(f"{out}  ({len(page)} candidates)")

print("\nrank -> reward + path (for stamping):")
for i, (reward, png, prompt) in enumerate(rows, 1):
    print(f"{i:2d}. {reward:.3f}  {png}")
