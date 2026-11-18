import cv2
import numpy as np
from config import (
    SCRATCH_MIN_AREA, SCRATCH_MAX_AREA, SCRATCH_MIN_ASPECT,
    SCRATCH_DOG_SIGMA1, SCRATCH_DOG_SIGMA2,
    SCRATCH_OTSU_SCALE, SCRATCH_EDGE_MARGIN,
)


def _shape_ok(c):
    area = cv2.contourArea(c)
    if not (SCRATCH_MIN_AREA <= area <= SCRATCH_MAX_AREA):
        return False
    x, y, w, h = cv2.boundingRect(c)
    aspect = max(w, h) / max(min(w, h), 1)
    return aspect >= SCRATCH_MIN_ASPECT


def _ignore_border(mask, margin=SCRATCH_EDGE_MARGIN):
    h, w = mask.shape
    m = int(min(w, h) * margin)
    mask[:m, :] = 0; mask[-m:, :] = 0
    mask[:, :m] = 0; mask[:, -m:] = 0
    return mask


def _scratch_mask(gray):
    """Bouwt een binair masker van krasachtige structuren via DoG + Otsu."""
    b1 = cv2.GaussianBlur(gray, (0, 0), SCRATCH_DOG_SIGMA1)
    b2 = cv2.GaussianBlur(gray, (0, 0), SCRATCH_DOG_SIGMA2)
    dog = cv2.absdiff(b1, b2)

    # Otsu-drempel, geschaald voor extra gevoeligheid
    otsu_val, _ = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(otsu_val * SCRATCH_OTSU_SCALE)
    _, th = cv2.threshold(dog, thr, 255, cv2.THRESH_BINARY)
    th = _ignore_border(th)
    return th


def detect_scratches_single(glare_image):
    """Basis: krassen op één strijklichtfoto."""
    gray = cv2.cvtColor(glare_image, cv2.COLOR_BGR2GRAY)
    mask = _scratch_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scratches = [c for c in contours if _shape_ok(c)]
    return scratches, mask


def detect_scratches_dual(glare_a, glare_b, match_dist=6):
    """
    Dual-glare: krassen moeten in BEIDE opnames op ongeveer dezelfde
    locatie liggen. Filtert reflecties weg die enkel in één opname zitten.
    """
    gray_a = cv2.cvtColor(glare_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(glare_b, cv2.COLOR_BGR2GRAY)

    mask_a = _scratch_mask(gray_a)
    mask_b = _scratch_mask(gray_b)

    # Dilateer mask_b licht zodat kleine uitlijningsfouten getolereerd worden
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * match_dist + 1, 2 * match_dist + 1))
    mask_b_dil = cv2.dilate(mask_b, kernel)

    # Alleen krassen die in A én in de buurt in B voorkomen
    mask_dual = cv2.bitwise_and(mask_a, mask_b_dil)

    contours, _ = cv2.findContours(mask_dual, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    scratches = [c for c in contours if _shape_ok(c)]
    return scratches, mask_dual


def draw_scratches(base_image, scratches, mask=None):
    """Rode contour om elke kras, optioneel masker als achtergrond."""
    out = base_image.copy()
    if mask is not None:
        overlay = out.copy()
        overlay[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    cv2.drawContours(out, scratches, -1, (0, 0, 255), 2)
    return out