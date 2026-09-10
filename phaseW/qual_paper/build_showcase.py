#!/usr/bin/env python3
"""Cherry-picked 28-step samples from the projector-reward student (checkpoint_avg_last5.pt,
phaseS4_CD_dinop_hard-rewRi-s16_s0_137621), no guidance."""
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont

CELL, PAD, LABEL_H = 340, 10, 60
ROOT = Path("phaseW/qual_paper/gen5/images/STUDENT")
PICKS = [
    (77, "a red stop sign and a white line"),
    (624, "wooden pencils and a leather sofa"),
    (1846, "The gentle, rolling hills of the countryside were a peaceful escape from the hustle and bustle of the city."),
    (1985, "The vibrant, glittering lights of the carnival rides spun and twirled in dizzying circles, thrilling and delighting the adventurous."),
    (574, "a triangular slice of bread and a cylindrical loaf"),
    (337, "a circular pendant light and a triangular wall hook"),
]


def font(sz, bold=False):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
              "/usr/share/fonts/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ncol = 3
    nrow = (len(PICKS) + ncol - 1) // ncol
    W = ncol * (CELL + PAD) + PAD
    H = nrow * (CELL + LABEL_H + PAD) + PAD + 40
    sheet = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(sheet)
    f_ti, f_pr = font(17, True), font(13)
    d.text((PAD, 10), "Cherry-picked 28-step samples, projector-reward student, w=1, no candidate generation", fill=(40, 60, 120), font=f_ti)
    for k, (idx, prompt) in enumerate(PICKS):
        r, c = divmod(k, ncol)
        x = PAD + c * (CELL + PAD)
        y = 40 + PAD + r * (CELL + LABEL_H + PAD)
        img_path = ROOT / f"p{idx:05d}" / "s28" / "cand0.png"
        im = Image.open(img_path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        sheet.paste(im, (x, y))
        d.rectangle([x - 1, y - 1, x + CELL, y + CELL], outline=(180, 180, 180), width=1)
        wrapped = textwrap.wrap(prompt, 46)[:3]
        for i, line in enumerate(wrapped):
            d.text((x, y + CELL + 6 + i * 16), line, fill=(20, 20, 20), font=f_pr)
    out = Path("phaseW/qual_paper/showcase_28step.png")
    sheet.save(out)
    print(f"wrote {out} ({W}x{H})")


if __name__ == "__main__":
    main()
