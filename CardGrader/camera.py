"""
Live camera-test voor de card grader (laptopwebcam of externe camera).

Toetsen in het venster:
  SPATIE  foto nemen en analyseren
  m       handmatig 4 hoeken aanklikken op de laatste foto (als de automatische detectie faalt)
  q / Esc afsluiten

Gebruik:  python camera.py              (camera 0)
          python camera.py --cam 1 --width 1920 --height 1080
"""
import argparse
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

import preprocessing as P
from config import CARD_W, CARD_H
from main import analyze

CAPTURE_DIR = "captures"
LIVE_MAX_SIDE = 640           # live detectie op verkleind beeld (sneller)


def sharpness(frame):
    """Scherpte: variantie van de Laplacian (hoger = scherper)."""
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def sharpest(frames):
    """Kies de scherpste van een paar opeenvolgende frames (webcams hebben bewegingsonscherpte/focus-jacht)."""
    return max(frames, key=sharpness)


def guide_rect(w, h, height_frac=0.8):
    """Hulpkader met kaartverhouding, midden in beeld."""
    ch = int(h * height_frac)
    cw = int(ch * CARD_W / CARD_H)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, x0 + cw, y0 + ch


class LiveDetector(threading.Thread):
    """Draait de kaartdetectie op de achtergrond zodat het live beeld vloeiend blijft."""

    def __init__(self):
        super().__init__(daemon=True)
        self._frame = None
        self._lock = threading.Lock()
        self.result = None                      # (hoekpunten in oorspronkelijke frame-coördinaten, confidence, flatness)
        self._stop = threading.Event()

    def submit(self, frame):
        with self._lock:
            self._frame = frame

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.03)
                continue
            s = min(1.0, LIVE_MAX_SIDE / max(frame.shape[:2]))
            small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
            try:
                pts, conf, _ = P.find_corners(small)
                self.result = (pts / s, float(conf), float(P.LAST_INFO.get("flatness", 0.0)))
            except Exception:
                self.result = None


def capture_and_analyze(frame, manual=False):
    """Slaat het frame op en draait de volledige analyse. Retourneert (report, pad)."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CAPTURE_DIR, f"card_{stamp}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return analyze(front_path=path, out_prefix=f"card_{stamp}"), path


def result_panel(report, summary_path):
    """Samenvattingsbeeld + scores naast elkaar."""
    summ = cv2.imread(summary_path)
    panel = np.full((summ.shape[0], 420, 3), 30, np.uint8)
    lines = [("Resultaat (schatting)", 0.8, (255, 255, 255))]
    sc = report["scores_0_10"]
    lines += [(f"Centrering   {sc['centering']}/10", 0.7, (200, 255, 200)),
              (f"  L/R {report['centering']['lr_ratio']} / {round(100 - report['centering']['lr_ratio'], 1)}"
               f"   T/B {report['centering']['tb_ratio']} / {round(100 - report['centering']['tb_ratio'], 1)}", 0.55, (200, 200, 200)),
              (f"Witte randen {sc['white_edges']}/10", 0.7, (200, 255, 200))]
    if report["white_edges"].get("reliable") is False:
        lines.append(("  ! lichte achtergrond: onbetrouwbaar", 0.55, (80, 160, 255)))
    if report.get("geometry") and not report["geometry"]["flat_ok"]:
        lines.append(("  ! kaart niet plat: minder betrouwbaar", 0.55, (80, 160, 255)))
    lines.append(("Krassen: geen glare-foto's", 0.6, (180, 180, 180)))
    lines.append((f"TOTAAL {report['overall_0_10']}/10", 0.9, (255, 255, 255)))
    y = 40
    for text, scale, color in lines:
        cv2.putText(panel, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(46 * max(scale, 0.6))
    return np.hstack([summ, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Camera {args.cam} kon niet geopend worden.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)           # werkt niet op elke webcam; zet focus anders vast in de camera-app
    print("Resolutie:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    det = LiveDetector()
    det.start()
    last_frame, status, busy = None, "", False
    last_submit = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Geen beeld van de camera.")
            break
        last_frame = frame
        if time.time() - last_submit > 0.5:
            det.submit(frame.copy())
            last_submit = time.time()

        view = frame.copy()
        h, w = view.shape[:2]
        x0, y0, x1, y1 = guide_rect(w, h)
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 255, 255), 1)

        info = [f"scherpte {sharpness(frame):.0f}"]
        r = det.result
        if r is not None:
            pts, conf, flat = r
            good = conf >= P.MIN_CONFIDENCE
            cv2.polylines(view, [pts.astype(np.int32)], True, (0, 255, 0) if good else (0, 0, 255), 3)
            info.append(f"detectie {conf:.2f} {'OK' if good else 'onzeker'}")
            if flat > 0.008:
                info.append("kaart niet plat?")
        info.append("SPATIE = analyseren | m = handmatig | q = stop")
        for i, t in enumerate(info):
            cv2.putText(view, t, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, t, (15, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        if status:
            cv2.putText(view, status, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Card grader", view)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord(" "):
            # kies de scherpste van 5 opeenvolgende frames
            frames = [frame] + [cap.read()[1] for _ in range(4)]
            best = sharpest([f for f in frames if f is not None])
            status = "Analyseren... (kan 10 s duren)"
            cv2.putText(view, status, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Card grader", view); cv2.waitKey(1)
            report, path = capture_and_analyze(best)
            if report is None:
                status = "Kaart niet gevonden - probeer opnieuw of druk 'm'"
                last_frame = best
            else:
                summary = os.path.join("debug", os.path.splitext(os.path.basename(path))[0] + "_summary.jpg")
                cv2.imshow("Resultaat", result_panel(report, summary))
                status = f"Klaar: {report['overall_0_10']}/10  ({path})"
        elif k == ord("m") and last_frame is not None:
            corners = P.manual_corners(last_frame)
            if corners is not None:
                card = P.warp_card(last_frame, corners)
                os.makedirs(CAPTURE_DIR, exist_ok=True)
                p = os.path.join(CAPTURE_DIR, "manual_warp.jpg")
                cv2.imwrite(p, card)
                status = f"Handmatige warp opgeslagen: {p}"

    det.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
