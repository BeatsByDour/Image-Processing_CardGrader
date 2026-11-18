"""
Fase A: kaartdetectie + perspective transform.

Aanpak (onafhankelijk van achtergrondkleur):
 1. Uniforme balken langs de beeldrand (bv. witte/zwarte opvulling) wegsnijden.
 2. Grof zoeken naar een rechthoek met kaartverhouding 63:88 waarvan de omtrek de sterkste
    randgradiënt heeft (coarse-to-fine, hoek -20..+20 graden).
 3. Per zijde verfijnen: meetpunten stemmen (mini-Hough over positie + kleine hoek) op een
    randlijn; de BUITENSTE goed ondersteunde lijn wordt gekozen, zodat de kaartrand
    wordt gevonden en niet de binnenrand van een holo/gekleurde rand. Daarna lijnfit.
 4. Vier lijnen snijden -> vier hoekpunten -> perspective transform.
"""
import cv2
import numpy as np
from config import CARD_W, CARD_H


CARD_ASPECT = 63 / 88          # breedte / hoogte van een TCG-kaart
SEARCH_SIZE = 500               # werkresolutie voor zoeken
MIN_FRAC = 0.5                  # kaart is minstens 50% van de lange zijde van de foto
SIZE_BONUS = 60.0
MIN_EDGE = 30.0
LAST_INFO = {}                  # info over de laatste detectie: confidence, flatness (afwijking/kaartbreedte)
EARLY_ACCEPT = 0.9
MIN_CONFIDENCE = 0.75           # onder deze randondersteuning beschouwen we de detectie als mislukt
EDGE_SPAN = 3
ASPECT_TOL = 0.035
EDGE_CAP = 90.0
RING_PENALTY = 0.8              # de ring net buiten de kaart mag geen rand bevatten
RING_FRAC = 0.02
CORNER_SKIP = 0.14              # negeer afgeronde hoeken (fractie van zijdelengte)


def _grad_maps(small):
    """Gradient per kleurkanaal (Gx, Gy) -> lijst van 3 paren."""
    s = cv2.GaussianBlur(small, (0, 0), 1.2).astype(np.float32)
    gx = [cv2.Sobel(s[:, :, c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)]
    gy = [cv2.Sobel(s[:, :, c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)]
    return gx, gy


def _proj_map(gx, gy, theta):
    """Gradient-sterkte loodrecht op een lijn met richting theta (max over kanalen), licht verbreed."""
    # normaal op richting (cos t, sin t)  ->  (-sin t, cos t)
    nx, ny = -np.sin(theta), np.cos(theta)
    p = np.max([np.abs(nx * a + ny * b) for a, b in zip(gx, gy)], axis=0)
    p = np.minimum(p, EDGE_CAP)
    return cv2.dilate(p, np.ones((7, 7), np.uint8))


def _side_points(W, H, n=60):
    """Meetpunten op de 4 zijden van een rechthoek rond (0,0), hoeken uitgesloten.
    Retourneert (pts_h, pts_v): bovenste/onderste zijde en linker/rechter zijde."""
    t = np.linspace(CORNER_SKIP, 1 - CORNER_SKIP, n)
    xs = (t - 0.5) * W
    ys = (t - 0.5) * H
    top = np.stack([xs, np.full(n, -H / 2)], 1)
    bot = np.stack([xs, np.full(n, H / 2)], 1)
    lef = np.stack([np.full(n, -W / 2), ys], 1)
    rig = np.stack([np.full(n, W / 2), ys], 1)
    return np.vstack([top, bot]), np.vstack([lef, rig])


def _score(maps, centers, W, H, theta, shape, outside_val=0.0, out_pen=3.0):
    """Score voor een set middelpunten (M,2) bij gegeven W,H,theta. Hoger = beter."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    ph, pv = _side_points(W, H)
    ph = ph @ R.T
    pv = pv @ R.T
    mh, mv = maps
    h, w = shape
    out = np.zeros(len(centers))
    for pts, m in ((ph, mh), (pv, mv)):
        xy = centers[:, None, :] + pts[None, :, :]
        xi = np.rint(xy[..., 0]).astype(int)
        yi = np.rint(xy[..., 1]).astype(int)
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        vals = np.where(inside, m[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], outside_val)
        # ongeveer de helft van de beste metingen telt: robuust voor een zwakke zijde
        out += np.sort(vals, axis=1)[:, vals.shape[1] // 4:].mean(axis=1)
        out -= out_pen * (~inside).mean(axis=1)
    return out


def _ridge_score(maps, centers, W, H, theta, shape):
    d = RING_FRAC * max(W, H)
    s0 = _score(maps, centers, W, H, theta, shape)
    so = _score(maps, centers, W + 2 * d, H + 2 * d, theta, shape, 0.0, 0.0)
    return s0 - RING_PENALTY * so


def _find_rect(img):
    h0, w0 = img.shape[:2]
    sc = SEARCH_SIZE / max(h0, w0)
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    gx, gy = _grad_maps(small)

    cache = {}
    def maps_for(theta):
        key = round(theta, 4)
        if key not in cache:
            cache[key] = (_proj_map(gx, gy, theta), _proj_map(gx, gy, theta + np.pi / 2))
        return cache[key]

    # --- grof zoeken ---
    gxs = np.arange(0.2 * w, 0.8 * w + 1, max(4, w * 0.035))
    gys = np.arange(0.2 * h, 0.8 * h + 1, max(4, h * 0.035))
    centers = np.array([(x, y) for y in gys for x in gxs])
    best = (-1e9, None)
    Hs = np.linspace(MIN_FRAC * max(h, w), 1.05 * max(h, w), 10)
    for th_deg in np.arange(-20, 20.1, 1.0):
        th = np.radians(th_deg)
        m = maps_for(th)
        for Hc in Hs:
            Wc = Hc * CARD_ASPECT
            sc_ = _score(m, centers, Wc, Hc, th, (h, w)) + SIZE_BONUS * Hc / max(h, w)
            i = int(np.argmax(sc_))
            if sc_[i] > best[0]:
                best = (sc_[i], (centers[i, 0], centers[i, 1], Wc, Hc, th))

    # --- fijn zoeken (coordinate descent) ---
    cx, cy, Wc, Hc, th = best[1]
    step = [3.0, 3.0, 3.0, 3.0, np.radians(1.0)]
    cur = best[0]
    for _ in range(40):
        improved = False
        for k in range(5):
            for sgn in (+1, -1):
                p = [cx, cy, Wc, Hc, th]
                p[k] += sgn * step[k]
                if not (CARD_ASPECT - ASPECT_TOL <= p[2] / p[3] <= CARD_ASPECT + ASPECT_TOL):
                    continue
                m = maps_for(p[4])
                v = _score(m, np.array([[p[0], p[1]]]), p[2], p[3], p[4], (h, w))[0] + SIZE_BONUS * p[3] / max(h, w)
                if v > cur + 1e-6:
                    cur = v; cx, cy, Wc, Hc, th = p; improved = True
        if not improved:
            step = [s_ * 0.5 for s_ in step]
            if step[0] < 0.4: break
    return (cx / sc, cy / sc, Wc / sc, Hc / sc, th), cur


def _refine_side_lines(img, rect, band_frac=0.03):
    """Verfijnt elke zijde: 80 meetpunten stemmen op de positie van een randlijn; kies de
    BUITENSTE goed ondersteunde lijn (kaartrand i.p.v. holo-binnenrand) en fit daarop."""
    cx, cy, W, H, th = rect
    blur = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    u = np.array([c, s]); v = np.array([-s, c])
    center = np.array([cx, cy])
    band_out = band_frac * max(W, H)
    band_in = 1.5 * band_out
    offs = np.arange(-band_in, band_in + band_out + 1, 1.0)        # sampled relative aan voorspelling, in 'buitenwaartse' richting
    nS = 80
    lines = []
    supports = []
    devs = []
    for axis_dir, normal, length, half_off in ((u, v, W, H / 2), (u, v, W, -H / 2),
                                               (v, u, H, W / 2), (v, u, H, -W / 2)):
        sgn = 1.0 if half_off > 0 else -1.0
        out_dir = normal * sgn
        samples = []     # per meetpunt: lijst van (t_pos, offset) pieken
        for t in np.linspace(-0.5 + CORNER_SKIP, 0.5 - CORNER_SKIP, nS):
            base = center + axis_dir * t * length + normal * half_off
            o = offs - band_in                                        # van -band_in (binnen) tot +band_out (buiten)
            samp = base[None, :] + o[:, None] * out_dir[None, :]
            mx = np.clip(samp[:, 0], 0, img.shape[1] - 1.001).astype(np.float32).reshape(-1, 1)
            my = np.clip(samp[:, 1], 0, img.shape[0] - 1.001).astype(np.float32).reshape(-1, 1)
            vals = np.stack([cv2.remap(blur[:, :, k], mx, my, cv2.INTER_LINEAR)[:, 0] for k in range(3)], 1)
            K = EDGE_SPAN
            d = np.zeros(len(vals))
            d[K:-K] = np.abs(vals[2 * K:] - vals[:-2 * K]).max(axis=1)   # verschil over 2K px: ook zachte randen
            pos = o + 0.0
            thr = max(MIN_EDGE, 0.3 * d.max())
            pk = [i for i in range(1, len(d) - 1) if d[i] >= thr and d[i] >= d[i - 1] and d[i] >= d[i + 1]]
            samples.append([(t, pos[i]) for i in pk])
        # mini-Hough per zijde: stem over (hoek, positie); kies hoek met sterkste lijn, dan de buitenste goed ondersteunde piek
        edges_ = np.arange(-band_in - 1, band_out + 3, 2.0)
        flat = [(t * length, p) for sm, t in zip(samples, np.linspace(-0.5 + CORNER_SKIP, 0.5 - CORNER_SKIP, nS)) for (_, p) in sm]
        if len(flat) < 10:
            return None
        xs_ = np.array([f[0] for f in flat]); ps_ = np.array([f[1] for f in flat])
        best = None
        for dd in np.radians(np.arange(-2.0, 2.01, 0.25)):
            hist, _ = np.histogram(ps_ - xs_ * np.tan(dd) * sgn, edges_)
            hs = np.convolve(hist, [1, 1, 1], 'same')
            if best is None or hs.max() > best[0]:
                best = (hs.max(), dd, hs)
        _, dd, hs = best
        need = max(0.25 * nS, 0.35 * hs.max())
        cands = [i for i in range(len(hs)) if hs[i] >= need and hs[i] >= hs[max(i - 1, 0)] and hs[i] >= hs[min(i + 1, len(hs) - 1)]]
        if not cands:
            return None
        k = max(cands)                                                # buitenste
        centre_pos = (edges_[k] + edges_[k + 1]) / 2
        pts = []
        for sm, t in zip(samples, np.linspace(-0.5 + CORNER_SKIP, 0.5 - CORNER_SKIP, nS)):
            x = t * length
            near = [p for (_, p) in sm if abs(p - x * np.tan(dd) * sgn - centre_pos) <= 3.5]
            if near:
                p = min(near, key=lambda q: abs(q - x * np.tan(dd) * sgn - centre_pos))
                pts.append(center + axis_dir * x + normal * half_off + out_dir * p)
        if len(pts) < 12:
            return None
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
        supports.append(len(pts) / nS)
        # vlakheid: hoe ver wijkt de randpositie per meetpunt af van de rechte lijn? (ruime tolerantie, 12 px)
        dev = []
        for sm, t in zip(samples, np.linspace(-0.5 + CORNER_SKIP, 0.5 - CORNER_SKIP, nS)):
            x = t * length
            expected = centre_pos + x * np.tan(dd) * sgn
            cand = [q_ - expected for (_, q_) in sm if abs(q_ - expected) <= 12]
            if cand:
                dev.append(min(cand, key=abs))
        devs.append(float(np.percentile(np.abs(dev), 90)) if len(dev) >= 10 else 12.0)
    return lines, float(np.mean(supports)), float(max(devs))


def _intersect(l1, l2):
    p1, d1 = l1; p2, d2 = l2
    A = np.array([d1, -d2]).T
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def trim_uniform_borders(img, std_max=1.5, min_frac=0.06, extra=5):
    """Snijdt echte foto-balken weg: PERFECT egale (std < 1.5) witte of zwarte rijen/kolommen langs de
    beeldrand (digitale opvulling, >= min_frac van de afmeting), plus een kleine marge tegen JPEG-ringing.
    Een gewone achtergrond (ook een egale, matte) heeft sensorruis en/of een andere kleur en wordt NIET
    weggesneden; anders zou de marge in de kaartrand zelf snijden."""
    h, w = img.shape[:2]
    g = img.astype(np.float32).mean(axis=2)
    colstd, rowstd = g.std(axis=0), g.std(axis=1)
    colmean, rowmean = g.mean(axis=0), g.mean(axis=1)

    def run(std, mean):
        i = 0
        while i < len(std) // 3 and std[i] < std_max and (mean[i] > 240 or mean[i] < 8):
            i += 1
        return i

    def cut(n, total):
        return n + extra if n >= min_frac * total else 0

    l, r = cut(run(colstd, colmean), w), cut(run(colstd[::-1], colmean[::-1]), w)
    t, b = cut(run(rowstd, rowmean), h), cut(run(rowstd[::-1], rowmean[::-1]), h)
    return img[t:h - b, l:w - r], (l, t)


def find_corners(img_full):
    img, (ox, oy) = trim_uniform_borders(img_full)
    pts, score, rect, flat = _find_corners_core(img)
    LAST_INFO.update(confidence=float(score), flatness=float(flat))
    return pts + np.array([ox, oy], np.float32), score, rect


def _color_rect(img):
    """Kandidaat 2: kaart = alles wat niet op de achtergrond (randstrook, k-means in Lab) lijkt."""
    h0, w0 = img.shape[:2]
    sc = 500 / max(h0, w0)
    sm = cv2.GaussianBlur(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), (5, 5), 0)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    b = max(3, int(0.02 * min(H, W)))
    strip = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                            lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    cv2.setRNGSeed(0)                # deterministische k-means
    _, _, cent = cv2.kmeans(strip, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                            3, cv2.KMEANS_PP_CENTERS)
    d = np.min([np.linalg.norm(lab - c, axis=2) for c in cent], axis=0)
    _, m = cv2.threshold(np.clip(d * 5, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.1 * H * W:
        return None
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw > rh:                      # zorg dat 'W' de korte zijde is (portret)
        rw, rh, ang = rh, rw, ang + 90
    th = np.radians(ang)
    th = (th + np.pi / 2) % np.pi - np.pi / 2
    return (cx / sc, cy / sc, rw / sc, rh / sc, th)


def _refine_rect(img, rect):
    """3 verfijningsrondes (smaller wordende band). Retourneert (hoekpunten, ondersteuning 0..1) of None."""
    pts, support, flat = None, 0.0, 0.0
    for _i, band in enumerate((0.05, 0.025, 0.012, 0.006)):
        out = _refine_side_lines(img, rect, band)
        if out is None:
            return None if pts is None else (pts, support, flat)
        lines, support, dev = out
        bot, top, rig, lef = lines
        pts = np.array([_intersect(top, lef), _intersect(top, rig), _intersect(bot, rig), _intersect(bot, lef)], np.float32)
        c_ = pts.mean(0)
        W = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
        H = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
        th = np.arctan2(*(pts[1] - pts[0])[::-1])
        rect = (c_[0], c_[1], W, H, th)
    # Vlakheid: een aparte meting rond de EINDrechthoek (rechthoek zit nu goed), met ruime band (2%)
    if pts is not None:
        out = _refine_side_lines(img, rect, 0.02)
        if out is not None:
            flat = out[2]
    return pts, support, flat


def _find_corners_core(img):
    """Probeert meerdere startkandidaten, verfijnt ze en kiest die met de meeste randondersteuning."""
    grad_rect, score = _find_rect(img)

    def candidates():
        yield grad_rect
        try:
            cr = _color_rect(img)
        except cv2.error:
            cr = None
        if cr is not None:
            yield cr

    best = None
    for rect in candidates():
        if best is not None and best[0] >= EARLY_ACCEPT:      # eerste kandidaat was al overtuigend
            break
        out = _refine_rect(img, rect)
        if out is None:
            continue
        pts, support, flat = out
        w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
        h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
        asp_err = abs(w / max(h, 1) - CARD_ASPECT)
        q = support - 3.0 * max(0.0, asp_err - 0.03)           # straf een afwijkende kaartverhouding
        if best is None or q > best[0]:
            best = (q, pts, rect, flat / max(w, 1))
    if best is None:
        cx, cy, W, H, th = grad_rect
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        pts = (np.array([[-W/2, -H/2], [W/2, -H/2], [W/2, H/2], [-W/2, H/2]]) @ R.T) + [cx, cy]
        return pts.astype(np.float32), 0.0, grad_rect, 1.0
    return best[1], best[0], best[2], best[3]


def order_corners(pts):
    """Sorteert 4 punten als: linksboven, rechtsboven, rechtsonder, linksonder."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    dd = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(dd)],
                     pts[np.argmax(s)], pts[np.argmax(dd)]], dtype=np.float32)


def warp_card(image, corners):
    """Perspective transform naar vaste CARD_W x CARD_H."""
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)


def find_card_corners(image):
    """Retourneert 4 hoekpunten (LB, RB, RO, LO) of None als de detectie onbetrouwbaar is."""
    try:
        pts, conf, _ = find_corners(image)
    except Exception:
        return None
    if conf < MIN_CONFIDENCE:
        return None
    pts = order_corners(pts)
    # sanity check: verhouding moet op een kaart lijken
    w = (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2
    h = (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2
    if h <= 0 or abs(w / h - CARD_ASPECT) > 0.08:
        return None
    return pts


def manual_corners(image, window="Klik 4 hoeken (LB, RB, RO, LO) - Esc = annuleren"):
    """Fallback: de gebruiker klikt zelf de 4 kaarthoeken aan. Retourneert hoekpunten of None."""
    h, w = image.shape[:2]
    scale = min(1.0, 900 / max(h, w))
    view = cv2.resize(image, None, fx=scale, fy=scale)
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))
            cv2.circle(view, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window, view)

    cv2.imshow(window, view)
    cv2.setMouseCallback(window, on_mouse)
    while len(pts) < 4:
        if cv2.waitKey(30) == 27:
            cv2.destroyWindow(window)
            return None
    cv2.destroyWindow(window)
    return order_corners(np.array(pts, np.float32))


def normalize_card(image_path, allow_manual=False):
    """Volledige Fase A: laad, detecteer, warp. Retourneert (genormaliseerde kaart, hoekpunten)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    corners = find_card_corners(img)
    if corners is None and allow_manual:
        corners = manual_corners(img)
    if corners is None:
        raise RuntimeError("Kaartcontour niet betrouwbaar gevonden - controleer achtergrond/contrast "
                           "of gebruik handmatige hoekselectie.")
    return warp_card(img, corners), corners


def background_is_light(image, corners, v_min=170, s_max=45):
    """Controleert of de achtergrond rond de kaart licht/wit is (dan is witte-randdetectie onbetrouwbaar,
    want witte slijtage is dan niet te onderscheiden van de achtergrond)."""
    image, (ox, oy) = trim_uniform_borders(image)          # foto-balken zijn geen achtergrond
    corners = corners - np.array([ox, oy], np.float32)
    h, w = image.shape[:2]
    quad = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(quad, corners.astype(np.int32), 255)
    k = max(5, int(0.02 * max(h, w)))
    outside = cv2.bitwise_not(cv2.dilate(quad, np.ones((k, k), np.uint8)))
    if int((outside > 0).sum()) < 0.01 * h * w:
        return False, None                      # (bijna) geen achtergrond zichtbaar
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = float(np.median(hsv[:, :, 2][outside > 0]))
    s = float(np.median(hsv[:, :, 1][outside > 0]))
    return (v > v_min and s < s_max), (round(v), round(s))
