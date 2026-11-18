import cv2
import numpy as np
from config import (
    CARD_W, CARD_H,
    CENTER_SCAN_FRAC, CENTER_INSET, CENTER_PERCENTILE, CENTER_PEAK_FRAC,
)


def _first_peak(profile, start, frac):
    """Eerste lokale piek (>= frac * max) in het profiel vanaf 'start'. None als er geen is."""
    seg = profile[start:]
    if len(seg) < 3:
        return None
    thr = max(frac * seg.max(), np.median(seg) * 2.0)
    for i in range(1, len(seg) - 1):
        if seg[i] >= thr and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
            return start + i + 2          # +2: midden van het 4-px verschilvenster
    return None


def _edge_profile(strip):
    """strip: (N, L, 3) Lab-profielen van de kaartrand naar binnen.
    Retourneert de randsterkte per diepte als LAAG percentiel over alle rijen/kolommen.
    Een echte rand (binnenkant van de print-rand) zit in (bijna) alle rijen; tekst of
    holo-patronen zitten maar in een deel -> die vallen weg bij een laag percentiel."""
    g = np.linalg.norm(strip[:, 4:] - strip[:, :-4], axis=2)
    return np.percentile(g, CENTER_PERCENTILE, axis=0)


def detect_print_border(card):
    """
    Meet de breedte van de printrand (kaartrand -> begin van het binnenkader) aan alle 4 zijden.
    Werkt zowel voor effen (gele) randen als voor holo/regenboog-randen, omdat we niet op
    absolute kleur drempelen maar op de positie van de randovergang.

    Retourneert (left, top, right, bottom) als MARGE-breedtes in pixels, of None bij mislukking.
    """
    h, w = card.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(card, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    n = int(CENTER_SCAN_FRAC * min(w, h))
    rows = np.arange(int(0.2 * h), int(0.8 * h), 2)
    cols = np.arange(int(0.2 * w), int(0.8 * w), 2)

    strips = {
        "left":   lab[rows, :n],
        "right":  lab[rows, ::-1][:, :n],
        "top":    lab[:n, cols].transpose(1, 0, 2),
        "bottom": lab[::-1][:n, cols].transpose(1, 0, 2),
    }
    res = {k: _first_peak(_edge_profile(s), CENTER_INSET, CENTER_PEAK_FRAC) for k, s in strips.items()}
    if any(v is None for v in res.values()):
        return None
    return res["left"], res["top"], res["right"], res["bottom"]


def compute_centering(card):
    border = detect_print_border(card)
    if border is None:
        # Detectie mislukt: markeer expliciet (report.py geeft dan 0 punten i.p.v. 10).
        return {"left": 0, "right": 0, "top": 0, "bottom": 0,
                "lr_ratio": 50.0, "tb_ratio": 50.0, "ok": False}

    left, top, right, bottom = border          # alle vier zijn marge-breedtes
    lr_total = left + right
    tb_total = top + bottom
    return {
        "left": int(left), "right": int(right),
        "top": int(top), "bottom": int(bottom),
        "lr_ratio": round(float(left / lr_total * 100), 1),   # aandeel linkermarge (50 = perfect)
        "tb_ratio": round(float(top / tb_total * 100), 1),
        "ok": True,
    }


def draw_centering(card, result):
    """Tekent het binnenkader (groen) en de gemeten randbreedtes."""
    out = card.copy()
    if not result.get("ok", True):
        cv2.putText(out, "centrering mislukt", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return out
    l, t, r, b = result["left"], result["top"], result["right"], result["bottom"]
    cv2.rectangle(out, (l, t), (CARD_W - 1 - r, CARD_H - 1 - b), (0, 255, 0), 2)
    return out