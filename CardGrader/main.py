import cv2
import os
import glob
from preprocessing import normalize_card, background_is_light, LAST_INFO
from config import FLATNESS_WARN
from centering import compute_centering, draw_centering
from edges import detect_white_edges, draw_white_edges
from scratches import (
    detect_scratches_single, detect_scratches_dual, draw_scratches,
)
from report import build_report, save_report, build_summary_image

SAMPLE_DIR = "samples"   # let op: hoofdletter S, zoals jouw map


def analyze(front_path, glare_a_path=None, glare_b_path=None, out_prefix="kaart"):
    print(f"\n--- Verwerken: {os.path.basename(front_path)} ---")

    # ---------- Fase A: normalisatie ----------
    try:
        card, corners = normalize_card(front_path)
    except Exception as e:
        print(f"  ⚠ Overgeslagen: {e}")
        return None

    flat = LAST_INFO.get("flatness", 0.0)
    flat_ok = flat <= FLATNESS_WARN
    if not flat_ok:
        print(f"  ⚠ Kaart lijkt niet plat (randafwijking {flat*100:.2f}% van de kaartbreedte): "
              "warp en centrering zijn minder betrouwbaar. Leg de kaart vlak (bv. onder een glasplaatje).")

    light_bg, bg_vs = background_is_light(cv2.imread(front_path), corners)
    if light_bg:
        print(f"  ⚠ Lichte achtergrond (V,S={bg_vs}): witte-randscore is onbetrouwbaar. "
              "Gebruik een donkere, matte achtergrond.")

    # ---------- Fase B: centrering ----------
    centering = compute_centering(card)
    centering_img = draw_centering(card, centering)
    print("  Centrering:", centering)

    # ---------- Fase C: witte randen ----------
    white_mask, white_per_side, white_boxes = detect_white_edges(card)
    white_img = draw_white_edges(card, white_mask, white_boxes)
    print("  Witte randen:", white_per_side)

    # ---------- Fase D: krassen (alleen als glare beschikbaar) ----------
    scratches = []
    scratch_img = card.copy()
    if glare_a_path and os.path.exists(glare_a_path):
        if glare_b_path and os.path.exists(glare_b_path):
            ga, _ = normalize_card(glare_a_path)
            gb, _ = normalize_card(glare_b_path)
            scratches, dual_mask = detect_scratches_dual(ga, gb)
            scratch_img = draw_scratches(ga, scratches, dual_mask)
            print(f"  Krassen (dual-glare): {len(scratches)}")
        else:
            ga, _ = normalize_card(glare_a_path)
            scratches, mask = detect_scratches_single(ga)
            scratch_img = draw_scratches(ga, scratches, mask)
            print(f"  Krassen (single-glare): {len(scratches)}")
    else:
        print("  Krassen: overgeslagen (geen glare-foto)")

    # ---------- Fase E: rapport ----------
    report = build_report(centering, white_per_side, scratches,
                          scratches_skipped=(glare_a_path is None))
    report["white_edges"]["reliable"] = not light_bg
    report["geometry"] = {"flatness_pct": round(flat * 100, 2), "flat_ok": bool(flat_ok)}

    json_p, txt_p = save_report(report, prefix=out_prefix)

    # ---------- Debug-afbeeldingen ----------
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite(f"debug/{out_prefix}_card.jpg", card)
    cv2.imwrite(f"debug/{out_prefix}_centering.jpg", centering_img)
    cv2.imwrite(f"debug/{out_prefix}_white.jpg", white_img)
    if glare_a_path:
        cv2.imwrite(f"debug/{out_prefix}_scratches.jpg", scratch_img)

    summary = build_summary_image(card, centering_img, white_img, scratch_img)
    cv2.imwrite(f"debug/{out_prefix}_summary.jpg", summary)

    print(f"  Totaalscore: {report['overall_0_10']}/10  "
          f"(centrering {report['scores_0_10']['centering']}, "
          f"witte randen {report['scores_0_10']['white_edges']})")

    return report

def process_all_samples():
    """Verwerkt alle afbeeldingen in samples/ automatisch."""
    if not os.path.isdir(SAMPLE_DIR):
        print(f"Map '{SAMPLE_DIR}' niet gevonden.")
        return

    # Eén glob, en case-insensitive filter
    extensions = (".jpg", ".jpeg", ".png")
    files = [
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR)
        if f.lower().endswith(extensions)
    ]

    if not files:
        print(f"Geen afbeeldingen gevonden in '{SAMPLE_DIR}'.")
        return

    print(f"Gevonden: {len(files)} afbeelding(en) in '{SAMPLE_DIR}'")
    all_reports = []
    for path in sorted(files):
        prefix = os.path.splitext(os.path.basename(path))[0][:20]
        prefix = "".join(c if c.isalnum() else "_" for c in prefix)
        r = analyze(front_path=path, out_prefix=prefix)
        if r:
            r["file"] = os.path.basename(path)
            all_reports.append(r)

    if all_reports:
        print("\n=== SAMENVATTING ALLE KAARTEN ===")
        for r in all_reports:
            print(f"  {r['file'][:30]:30s}  ->  {r['overall_0_10']}/10")

if __name__ == "__main__":
    process_all_samples()