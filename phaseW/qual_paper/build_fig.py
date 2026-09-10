#!/usr/bin/env python3
"""Paper figure: two CompBench 2D-spatial prompts where the student places both named objects
correctly and the frozen 28-step CFG-7 teacher does not, at identical initial noise. Manually
verified by inspection, not selected by score alone (see phaseW/qual_paper/NOTES.md)."""
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont

CELL, PAD, LABEL_W, HEAD_H = 300, 8, 260, 60
ROOT = Path("phaseW/qual_paper")
ROWS = [
    (1048, "a boy on the left of a balloon", "left/right"),
    (1143, "a candle on the top of a chicken", "top/bottom"),
]
COLS = [
    ("Student, 4 steps\n(no guidance)", lambda i: ROOT / f"gen2/images/STUDENT/p{i:05d}/s4/cand0.png"),
    ("Student, 8 steps\n(no guidance)", lambda i: ROOT / f"gen3/images/STUDENT/p{i:05d}/s8/cand0.png"),
    ("Student, 28 steps\n(no guidance)", lambda i: ROOT / f"gen3/images/STUDENT/p{i:05d}/s28/cand0.png"),
    ("Frozen teacher, 28 steps\n(CFG, w=7)", lambda i: ROOT / f"gen2/images/TEACHER/p{i:05d}/s28/cand0.png"),
]


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def cell(path):
    if path.exists():
        return Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
    im = Image.new("RGB", (CELL, CELL), (40, 40, 44))
    ImageDraw.Draw(im).text((10, CELL // 2), "missing", fill=(200, 90, 90), font=font(16))
    return im


def main():
    ncol, nrow = len(COLS), len(ROWS)
    W = LABEL_W + ncol * (CELL + PAD) + PAD
    H = HEAD_H + nrow * (CELL + PAD) + PAD
    sheet = Image.new("RGB", (W, H), (252, 252, 250))
    d = ImageDraw.Draw(sheet)
    f_hd, f_pr = font(16, True), font(15)
    for c, (name, _) in enumerate(COLS):
        x = LABEL_W + c * (CELL + PAD)
        col = (20, 110, 40) if c < 3 else (140, 40, 40)
        for k, line in enumerate(name.split("\n")):
            d.text((x + 6, 8 + k * 19), line, fill=col, font=f_hd)
    for r, (idx, prompt, kind) in enumerate(ROWS):
        y = HEAD_H + r * (CELL + PAD)
        wrapped = textwrap.wrap(f'"{prompt}"', 20)
        for k, line in enumerate(wrapped):
            d.text((PAD, y + 8 + k * 19), line, fill=(15, 15, 15), font=f_pr)
        for c, (_, pathfn) in enumerate(COLS):
            x = LABEL_W + c * (CELL + PAD)
            sheet.paste(cell(pathfn(idx)), (x, y))
            if c < 3:
                d.rectangle([x - 2, y - 2, x + CELL + 1, y + CELL + 1], outline=(20, 110, 40), width=3)
            else:
                d.rectangle([x - 2, y - 2, x + CELL + 1, y + CELL + 1], outline=(140, 40, 40), width=3)
    out = ROOT / "qual_spatial_paper.png"
    sheet.save(out)
    sheet.convert("RGB").save(ROOT / "qual_spatial_paper.pdf")
    print(f"wrote {out} and .pdf ({W}x{H})")


if __name__ == "__main__":
    main()
