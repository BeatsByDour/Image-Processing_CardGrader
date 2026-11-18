# Vaste kaartafmetingen na perspective transform (verhouding ± 63:88)
CARD_W = 750
CARD_H = 1050

# Centrering (meting van de printrand via randovergang-profiel)
CENTER_SCAN_FRAC = 0.14        # zoek de rand binnen 14% van de korte zijde
CENTER_INSET = 5               # negeer de buitenste px (uitlijnfout van de warp)
CENTER_PERCENTILE = 25         # laag percentiel: tekst/holo-patronen vallen weg
CENTER_PEAK_FRAC = 0.5         # pieksterkte t.o.v. max in het zoekvenster

# Randzone voor witdetectie: smalle strook langs de buitenrand (witte slijtage zit op de rand)
BORDER_MARGIN = 0.025          # breedte randzone als fractie van de korte zijde (~19 px)
CARD_CORNER_RADIUS = 0.04      # afronding van de kaarthoeken (fractie van de breedte)
WHITE_OUTER_TOUCH = 4          # een witte vlek moet de buitenste 4 px van de kaartrand raken (anders: onderdeel van de print)
EDGE_INSET = 1                 # buitenste px negeren (uitlijnfout warp / achtergrondrand)

# Witdetectie (HSV)
WHITE_S_MAX = 30
WHITE_V_MIN = 205
WHITE_MIN_AREA = 25           # minimale oppervlakte witvlek (px)
WHITE_CONTRAST = 22           # wit moet zoveel helderder zijn dan de randkleur net erachter (V)
WHITE_REF_DEPTH = (12, 26)    # diepte (px vanaf de rand) waar de referentie-randkleur gemeten wordt

# Krassen / deuken
SCRATCH_MIN_AREA = 15
SCRATCH_MAX_AREA = 5000
SCRATCH_MIN_ASPECT = 2.5
SCRATCH_DOG_SIGMA1 = 1.0
SCRATCH_DOG_SIGMA2 = 3.0
SCRATCH_OTSU_SCALE = 0.85      # fractie van Otsu-drempel (lager = gevoeliger)
SCRATCH_EDGE_MARGIN = 0.02     # negeer pixels vlak langs de kaartrand

# Rapport
REPORT_DIR = "reports"
# Vlakheid van de kaart (afwijking van de randen t.o.v. een rechte lijn, als fractie van de kaartbreedte)
FLATNESS_WARN = 0.008          # boven 0,8%: kaart gebogen/golvend -> warp en centrering minder betrouwbaar
