import os
import json
import cv2
import numpy as np
from config import REPORT_DIR


def _centering_score(lr_ratio, tb_ratio, centering_data):
    """
    Hoe dichter bij 50/50, hoe hoger de score.
    Als de meting mislukt is (alles 0), geven we een 0.0 in plaats van een 10.0.
    """
    # Vangnet: als de detectie faalde (geen randen gevonden)
    if not centering_data.get("ok", True) or (centering_data["left"] == 0 and centering_data["right"] == 0):
        return 0.0

    def side_score(r):
        dev = abs(r - 50.0)                 # 0 = perfect
        return max(0.0, 10.0 - (dev / 5.0) * 2.0)

    return round((side_score(lr_ratio) + side_score(tb_ratio)) / 2, 1)


def _white_score(per_side_pct):
    """Witte rand: 0% = 10, >=5% op een zijde = 0 (lineair)."""
    worst = max(per_side_pct.values()) if per_side_pct else 0.0
    return round(max(0.0, 10.0 - (worst / 5.0) * 10.0), 1)


def _scratch_score(n_scratches, total_scratch_area):
    """Vuistregel: 0 krassen = 10, daarna lineair omlaag."""
    penalty = min(10.0, n_scratches * 0.8 + total_scratch_area / 400.0)
    return round(max(0.0, 10.0 - penalty), 1)


def build_report(centering, white_per_side, scratches, scratches_skipped=False):
    # Krassen-totaaloppervlak
    scratch_area = sum(cv2.contourArea(c) for c in scratches)

    scores = {
        # We geven 'centering' nu mee aan de functie
        "centering": float(_centering_score(
            centering["lr_ratio"], 
            centering["tb_ratio"], 
            centering
        )),
        "white_edges": float(_white_score(white_per_side)),
    }
    if not scratches_skipped:
        scores["scratches"] = float(_scratch_score(len(scratches), scratch_area))

    # Overgeslagen onderdelen tellen niet mee (anders trekt een nep-10 het gemiddelde omhoog)
    overall = float(round(np.mean(list(scores.values())), 1))

    return {
        "centering": centering,
        "white_edges": {
            "per_side_pct": white_per_side,
            "worst_side": max(white_per_side, key=white_per_side.get)
                           if white_per_side else None,
        },
        "scratches": {
            "count": int(len(scratches)),
            "total_area": float(round(scratch_area, 1)),
            "skipped": bool(scratches_skipped),
        },
        "scores_0_10": scores,
        "overall_0_10": overall,
    }


def save_report(report, out_dir=REPORT_DIR, prefix="kaart"):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{prefix}_report.json")
    txt_path = os.path.join(out_dir, f"{prefix}_report.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== TCG Kaart Conditierapport ===\n\n")
        c = report["centering"]
        f.write(f"Centrering: L/R = {c['lr_ratio']} / "
                f"{round(100 - c['lr_ratio'], 1)}, "
                f"T/B = {c['tb_ratio']} / "
                f"{round(100 - c['tb_ratio'], 1)}\n")
        f.write(f"  -> score: {report['scores_0_10']['centering']}/10\n\n")

        w = report["white_edges"]
        f.write("Witte randen (per zijde %):\n")
        for side, val in w["per_side_pct"].items():
            f.write(f"  {side:7s}: {val}%\n")
        f.write(f"  -> score: {report['scores_0_10']['white_edges']}/10\n")
        if w.get("reliable") is False:
            f.write("  (onbetrouwbaar: lichte achtergrond, witte slijtage niet van achtergrond te scheiden)\n")
        f.write("\n")

        s = report["scratches"]
        if s.get("skipped"):
            f.write("Krassen: overgeslagen (geen glare-foto)\n\n")
        else:
            f.write(f"Krassen: {s['count']} gedetecteerd, "
                    f"totaaloppervlak = {s['total_area']} px\n")
            f.write(f"  -> score: {report['scores_0_10']['scratches']}/10\n\n")

        g = report.get("geometry")
        if g and not g["flat_ok"]:
            f.write(f"WAARSCHUWING: kaart niet plat (randafwijking {g['flatness_pct']}%); centrering/randen minder betrouwbaar\n\n")
        f.write(f"TOTAALSCORE: {report['overall_0_10']}/10\n")

    return json_path, txt_path


def build_summary_image(card, centering_img, white_img, scratch_img):
    """Maakt een 2x2 overzicht voor visuele inspectie."""
    from config import CARD_W, CARD_H
    tiles = [card, centering_img, white_img, scratch_img]
    tiles = [cv2.resize(t, (CARD_W // 2, CARD_H // 2)) for t in tiles]
    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top, bottom])