"""
loi_decrets_patterns.py
--------------------------
Patterns regex pour repérer les références légales dans le texte français
du Bulletin Officiel marocain.

Calibré sur des occurrences réelles trouvées dans les PDF du projet, ex :
    "dahir n° 1-09-20 du 22 safar 1430 (18 février 2009)"
    "dahir portant loi n° 1-73-255 du 27 chaoual 1393"
    "loi n° 03-25 relative aux organismes de placement collectif"
    "Décret n° 2-25-1062"
    "décret n°2-08-562"                              (sans espace avant le n°)
    "Arrêté conjoint du ministre de l'industrie et du commerce"
    "Bulletin officiel » n° 7499 du 25 chaoual 1447 (13 avril 2026)"

Approche : ces références ont un format numérique trop irrégulier (espaces
variables, "n°"/"n º"/"no", tirets) pour un EntityRuler à base de patterns
token-par-token spaCy fiable. On utilise donc des regex sur le texte brut,
puis on les convertit en spans spaCy via doc.char_span() dans
entity_ruler_builder_fr.py.

Chaque pattern retourne le texte SANS ambiguïté possible entre le numéro et
sa date associée (le groupe complet est capturé pour audit).
"""

import re

# Variantes de "n°" rencontrées dans les PDF réels (symbole °, "o", "º",
# avec ou sans espace, parfois collé au mot précédent)
_N = r"n[°oº]\s?"
_NUM = r"[\d]+(?:[-–.][\d]+){1,2}"  # Accepte 1-09-20, 1.09.20, 2-08-562

# Date textuelle après "du", ex "22 safar 1430" / "25 chaoual 1447" : jour
# (1-2 chiffres) + nom de mois (mot) + année (3-4 chiffres). Volontairement
# précis (pas un simple "tout sauf ponctuation") : un test réel montrait
# qu'une classe de caractères large avalait la phrase suivante en entier
# (et même la parenthèse de date grégorienne, cassant sa fermeture).
_DATE_TEXT = r"\d{1,2}\s+[^\d\s,;.\n()]+\s+\d{3,4}"

# Date entre parenthèses type "(18 février 2009)" ou hégirien "27 chaoual 1393"
_DATE_PARENS = r"(?:\s*\([^)]{4,40}\))?"

# Anticipe le début d'une nouvelle entité juridique après une virgule
# (ex: ", le décret" / ", la loi" — mais pas ", aux sociétés")
_NEW_ENTITY = r'(?:[;.]\s+|,\s*(?:[ld][eéa]|l\'|d\'|du|il|elle|ce)\s|\n)'

# --- DAHIR (y compris "dahir portant loi", "dahir relatif à ...") ---
DAHIR_PATTERN = re.compile(
    rf"[Dd]ahir"
    rf"(?:\s+portant\s+(?:loi\s+)?)?"  # " portant " ou " portant loi "
    rf"\s*{_N}{_NUM}"
    rf"(?:\s+du\s+{_DATE_TEXT}){{0,1}}{_DATE_PARENS}"
    rf"(?:\s+relative?\s+[àa](?:(?!{_NEW_ENTITY})[^;.\n]){{0,200}})?",
)

# --- LOI (numéro type "03-25", "12-06", parfois "1-73-255" pour les lois anciennes) ---
# Le texte "relative à ..." s'arrête devant un `, le/la/les/l'` qui signale une
# nouvelle entité, mais capture les virgules internes au titre.
# Accepte "loi organique n° 066-13" (qualificatif entre "loi" et "n°"),
# courant dans les lois constitutionnelles/organiques marocaines.
LOI_PATTERN = re.compile(
    rf"[Ll]oi\s+(?:(?:organique|cadre|ordinaire)\s+)?(?:{_N})?\s*{_NUM}"
    rf"(?:\s+relative?s?\s+[àa](?:(?!{_NEW_ENTITY})[^;.\n]){{0,200}})?",
)
# --- DECRET (numéro type "2-25-1062", "2-08-562") ---
DECRET_PATTERN = re.compile(
    rf"[Dd][ée]cret\s+(?:{_N})?\s*{_NUM}"
    rf"(?:\s+relatif\s+[àa](?:(?!{_NEW_ENTITY})[^;.\n]){{0,200}})?",
)

# --- ARRETE (simple ou conjoint, souvent sans numéro, identifié par le ministère) ---
# Accepte aussi "arrêté n° 1165-26 du 25 juin 2026" (numéro + date, sans
# ministère) et "arrêté susvisé n° 256-91" (renvoi à un arrêté antérieur).
ARRETE_PATTERN = re.compile(
    rf"[Aa]rr[êe]t[ée](?:\s+susvisé)?(?:\s+{_N}{_NUM})?"
    rf"(?:\s+conjoint)?"
    rf"(?:\s+du\s+ministre[^,;.\n]{{0,100}}|\s+du\s+{_DATE_TEXT})?",
)

# --- Référence au Bulletin Officiel lui-même (numéro + date de publication) ---
BULLETIN_OFFICIEL_PATTERN = re.compile(
    rf"Bulletin\s+officiel\s*»?\s*{_N}\d{{3,4}}(?:\s+du\s+{_DATE_TEXT}){{0,1}}{_DATE_PARENS}",
)

CIRCULAIRE_PATTERN = re.compile(
    rf"[Cc]irculaire\s+(?:{_N})?\s*[\d]+(?:[-–./][\d]+){{0,2}}",
)

# --- ORG (sociétés) ---
# Deux formes réelles du BO :
#   - "Société « ... »" (dénomination entre guillemets français) ;
#   - raisons sociales tout en majuscules sans accents, terminées par un
#     suffixe courant : "MURPHY MOROCCO OIL CO., LTD",
#     "CHARIOT OIL & GAS HOLDINGS", "TANGER MED PORT AUTHORITY".
# Les minuscules/accents exclus du run majuscule évitent de capturer les
# noms de ministères ("MINISTRE DE L'ÉCONOMIE" contient des é/accentués).
# Le tiret est autorisé (raisons sociales avec trait d'union, ex.
# "AL-MAGHRIB").
_ALLCAPS_NAME = r"[A-Z][A-Z0-9]*(?:\s+[A-Z0-9&'.+-]+)*"
ORG_PATTERN = re.compile(
    rf"(?:[Ss]oci[ée]t[ée]\s+«[^»]{{2,80}}»"
    rf"|\bBANK\s+AL-MAGHRIB\b"
    rf"|\b{_ALLCAPS_NAME}\s+(?:CO\.?\s*,\s*(?:LTD|INC|PLC|LLC)\b|CO\b"
    rf"|LTD\b"
    rf"|LLC\b|PLC\b|INC\b|SARL\b|SPA\b|S\.A\.\b|GROUP\b|HOLDINGS\b"
    rf"|PORT\s+AUTHORITY\b|BANK\b(?!\s+[A-Z])|AGENCY\b))",
)

# --- MONEY (montants) ---
# Formes réelles du BO (ex. convention de crédit CMA, BO_7522) :
#   - numérique : "100.000.000,00 euros", "2.500.000 DH" ;
#   - en toutes lettres : "cent millions d'euros" (souvent suivi du
#     montant numérique entre parenthèses).
# Chaîne BRUTE (pas rf"") : les quantifieurs doivent être en accolades
# SIMPLES — des accolades doublées (syntaxe f-string) feraient matcher le
# littéral "{1,3}" et le pattern ne matcherait JAMAIS (régression
# confirmée : "100.000.000,00 euros" non capturé).
_AMOUNT_NUM = r"\d{1,3}(?:[.\s]\d{3})+,\d{2}\s*(?:DH|MAD|EUR|USD|euros?|dirhams?|dollars?)"
_MAGNITUDE_WORDS = (r"(?:cent|deux cents|trois cents|cinq cents|"
                    r"deux|trois|quatre|cinq|six|sept|huit|neuf|dix|douze|quinze|"
                    r"vingt|trente|quarante|cinquante|soixante)")
MONEY_PATTERN = re.compile(
    rf"(?:{_AMOUNT_NUM}"
    rf"|\b{_MAGNITUDE_WORDS}\s+(?:millions?|milliards?)\s+d['’]?euros?)",
)

# Regroupement pour itération facile côté entity_ruler_builder_fr.py
# Note : MINISTERE n'est pas ici car il est géré par l'EntityRuler spaCy
# (patterns/fr/ministeres.jsonl) qui capture les noms complets, contrairement
# à une regex qui tronquerait à la première ponctuation.
LEGAL_REFERENCE_PATTERNS = {
    "DAHIR": DAHIR_PATTERN,
    "LOI": LOI_PATTERN,
    "DECRET": DECRET_PATTERN,
    "ARRETE": ARRETE_PATTERN,
    "BULLETIN_OFFICIEL": BULLETIN_OFFICIEL_PATTERN,
    "MONEY": MONEY_PATTERN,
    "ORG": ORG_PATTERN,
}


if __name__ == "__main__":
    sample = (
        "Vu le dahir n° 1-09-20 du 22 safar 1430 (18 février 2009) et le "
        "dahir portant loi n° 1-73-255 du 27 chaoual 1393, en application "
        "de la loi n° 03-25 relative aux organismes de placement collectif, "
        "le décret n°2-08-562 abroge l'arrêté du ministre de l'industrie et "
        "du commerce, publié au « Bulletin officiel » n° 7499 du 25 chaoual "
        "1447 (13 avril 2026)."
    )
    for label, pattern in LEGAL_REFERENCE_PATTERNS.items():
        matches = pattern.findall(sample)
        print(f"{label}: {matches}")