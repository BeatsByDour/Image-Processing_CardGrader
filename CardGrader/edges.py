import cv2
import numpy as np
from config import (
    CARD_W, CARD_H, BORDER_MARGIN, CARD_CORNER_RADIUS, EDGE_INSET, WHITE_OUTER_TOUCH,
    WHITE_S_MAX, WHITE_V_MIN, WHITE_MIN_AREA, WHITE_CONTRAST, WHITE_REF_DEPTH,
)


def _rounded_card_mask(h, w, inset=0):
    """Gevulde afgeronde rechthoek = het kaartoppervlak (zonder achtergrond in de hoeken)."""
    r = int(CARD_CORNER_RADIUS * w)
    x0, y0, x1, y1 = inset, inset, w - 1 - inset, h - 1 - inset
    r = max(1, r - inset)
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def _border_zone(h, w, margin):
    """Smalle ring langs de kaartrand: binnen het afgeronde kaartvlak, buiten het inwendige."""
    m = int(min(w, h) * margin)
    outer = _rounded_card_mask(h, w, inset=EDGE_INSET)
    inner = _rounded_card_mask(h, w, inset=EDGE_INSET + m)
    return cv2.subtract(outer, inner)


def _side_masks(h, w, margin):
    """Randzone opgesplitst in boven/onder/links/rechts. Hoeken gaan naar de dichtstbijzijnde zijde
    via de diagonalen, zodat elke randpixel precies bij 1 zijde hoort."""
    zone = _border_zone(h, w, margin)
    yy, xx = np.mgrid[0:h, 0:w]
    # genormaliseerde afstand tot elke zijde; kleinste wint
    d = np.stack([yy / h, (h - 1 - yy) / h, xx / w, (w - 1 - xx) / w])
    which = np.argmin(d, axis=0)
    out = {}
    for i, name in enumerate(("top", "bottom", "left", "right")):
        out[name] = ((which == i) & (zone > 0)).astype(np.uint8) * 255
    return out


def _reference_v(v, masks):
    """Per randpixel de typische helderheid van de randkleur NET achter de rand (dieper in de kaart).
    Holo-strepen lopen door tot daar en vallen dus niet op; echte witte slijtage wel."""
    h, w = v.shape
    d0, d1 = WHITE_REF_DEPTH
    vf = v.astype(np.float32)
    k = 15
    def smooth(a):
        pad = np.pad(a, k // 2, mode="edge")
        return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    ref_l = smooth(np.median(vf[:, d0:d1], axis=1))
    ref_r = smooth(np.median(vf[:, w - d1:w - d0], axis=1))
    ref_t = smooth(np.median(vf[d0:d1, :], axis=0))
    ref_b = smooth(np.median(vf[h - d1:h - d0, :], axis=0))
    ref = np.zeros((h, w), np.float32)
    ref[:, :] = 0
    ref[masks["left"] > 0] = np.broadcast_to(ref_l[:, None], (h, w))[masks["left"] > 0]
    ref[masks["right"] > 0] = np.broadcast_to(ref_r[:, None], (h, w))[masks["right"] > 0]
    ref[masks["top"] > 0] = np.broadcast_to(ref_t[None, :], (h, w))[masks["top"] > 0]
    ref[masks["bottom"] > 0] = np.broadcast_to(ref_b[None, :], (h, w))[masks["bottom"] > 0]
    return ref


def detect_white_edges(card, margin=BORDER_MARGIN):
    """
    Retourneert:
      - white_mask: binair beeld met witte slijtage in de randzone
      - per_side: dict met per zijde het percentage witte pixels in die randzone
      - boxes: lijst van (x, y, w, h) bounding boxes van gedetecteerde vlekken
    """
    h, w = card.shape[:2]
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    masks = _side_masks(h, w, margin)
    ref_v = _reference_v(v, masks)

    # 1. Wit = lage verzadiging EN hoge helderheid EN duidelijk helderder dan de randkleur erachter
    white = ((s < WHITE_S_MAX) & (v > WHITE_V_MIN) &
             (v.astype(np.float32) - ref_v > WHITE_CONTRAST)).astype(np.uint8) * 255

    # 2. Alleen de smalle randzone van het afgeronde kaartvlak (geen hoeken-achtergrond)
    border_zone = np.zeros((h, w), np.uint8)
    for m in masks.values():
        border_zone = cv2.bitwise_or(border_zone, m)
    white = cv2.bitwise_and(white, border_zone)

    # 3. Ruis wegfilteren
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 4. Vlekken filteren op oppervlakte + bounding boxes verzamelen
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(white)
    boxes = []
    # Buitenste strook van de kaartrand: echte witte slijtage begint hier; kenmerken van de print
    # (cirkel van de pre-evolutie, holo-strepen) raken de buitenrand niet.
    outer_ring = cv2.subtract(_rounded_card_mask(h, w, EDGE_INSET),
                              _rounded_card_mask(h, w, EDGE_INSET + WHITE_OUTER_TOUCH))
    for c in contours:
        if cv2.contourArea(c) < WHITE_MIN_AREA:
            continue
        blob = np.zeros_like(white)
        cv2.drawContours(blob, [c], -1, 255, -1)
        if not cv2.bitwise_and(blob, outer_ring).any():
            continue
        cv2.drawContours(clean, [c], -1, 255, -1)
        boxes.append(cv2.boundingRect(c))

    # 5. Score per zijde (percentage van de werkelijke randzone-oppervlakte)
    per_side = {}
    for side, m in masks.items():
        area = max(int((m > 0).sum()), 1)
        cnt = int((cv2.bitwise_and(clean, m) > 0).sum())
        per_side[side] = float(round(cnt / area * 100, 3))

    return clean, per_side, boxes


def draw_white_edges(card, white_mask, boxes):
    """Geannoteerde weergave: randzone (blauw), witte vlekken in rood omkaderd."""
    out = card.copy()
    h, w = card.shape[:2]
    zone = _border_zone(h, w, BORDER_MARGIN)
    cnts, _ = cv2.findContours(zone, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 128, 0), 1)
    overlay = out.copy()
    overlay[white_mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(out, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (0, 255, 255), 1)
    return out
