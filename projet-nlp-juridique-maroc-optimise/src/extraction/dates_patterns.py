"""
dates_patterns.py
-------------------
Extraction et normalisation des dates FR pour l'étape 3.

Deux calendriers cohabitent dans les Bulletins Officiels marocains :
  - Le calendrier hégirien (souvent transcrit en toutes lettres, ex.
    "22 safar 1430"), qui fait foi juridiquement.
  - Le calendrier grégorien correspondant, généralement donné entre
    parenthèses juste après (ex. "22 safar 1430 (18 février 2009)").

Les deux sont extraits séparément (labels DATE_HIJRI / DATE_GREGORIAN)
plutôt que fusionnés en une seule entité "date du dahir", pour rester
réutilisable : une date peut aussi apparaître seule, hors contexte de
dahir (ex. une date d'entrée en vigueur, un délai...).

Limite connue : l'orthographe française des mois hégiriens n'est PAS
standardisée dans les BO marocains (on trouve "hija"/"hijja"/"dou al
hijja", "kaada"/"kaada"/"dou al kaada", "joumada I"/"joumada al oula"...).
La liste MOIS_HIJRI_FR ci-dessous couvre les variantes les plus fréquentes
observées, mais n'est probablement pas exhaustive — à enrichir au fur et à
mesure des documents rencontrés (regarder les dates qui ne matchent pas
DATE_HIJRI_PATTERN pour repérer de nouvelles variantes).
"""

import re

MOIS_GREGORIEN_FR = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}

# Variantes de transcription française des mois hégiriens rencontrées dans
# les BO marocains (voir limite connue ci-dessus).
MOIS_HIJRI_FR = {
    "moharrem": 1, "mouharram": 1,
    "safar": 2,
    "rabii i": 3, "rabia i": 3, "rabii al oula": 3, "rabii premier": 3,
    "rabii 1": 3, "rabia 1": 3,
    "rabii ii": 4, "rabia ii": 4, "rabii al akhira": 4, "rabii ii chaabane": 4,
    "rabii 2": 4, "rabia 2": 4, "rabii 11": 4, "rabia 11": 4,
    "joumada i": 5, "joumada al oula": 5, "joumada premier": 5,
    "joumada 1": 5, "jourmada i": 5, "jourmada 1": 5,
    "joumada ii": 6, "joumada al akhira": 6, "joumada ii chaabane": 6,
    "joumada 2": 6, "joumada 11": 6, "jourmada it": 6, "jourmada ii": 6,
    "rejeb": 7, "rajab": 7,
    "chaabane": 8, "chaabana": 8,
    "ramadan": 9,
    "chaoual": 10, "chawwal": 10,
    "kaada": 11, "dou al kaada": 11, "kaâda": 11,
    "hija": 12, "hijja": 12, "dou al hijja": 12,
}

# Trie par longueur décroissante pour que les regex "rabii ii" soient
# essayées avant "rabii i" (sinon "rabii i" matcherait le début de "rabii
# ii" et tronquerait la capture du mois).
_HIJRI_MONTHS_SORTED = sorted(MOIS_HIJRI_FR.keys(), key=len, reverse=True)
_GREG_MONTHS_SORTED = sorted(MOIS_GREGORIEN_FR.keys(), key=len, reverse=True)

DATE_GREGORIAN_PATTERN = re.compile(
    rf"\b(\d{{1,2}})(?:er)?\s+({'|'.join(_GREG_MONTHS_SORTED)})\s+(\d{{4}})\b",
    re.IGNORECASE,
)

DATE_HIJRI_PATTERN = re.compile(
    rf"\b(\d{{1,2}})(?:er)?\s+({'|'.join(_HIJRI_MONTHS_SORTED)})\s+(\d{{3,4}})\b",
    re.IGNORECASE,
)


def extract_dates_fr(text: str):
    """
    Retourne une liste de LegalEntity (label DATE_GREGORIAN ou DATE_HIJRI),
    avec dans .meta les composants structurés {day, month, month_name,
    year, calendar}. Import différé de LegalEntity pour éviter un import
    circulaire (entities.py n'importe pas ce module).
    """
    from src.extraction.entities import LegalEntity

    found = []

    for match in DATE_GREGORIAN_PATTERN.finditer(text):
        day, month_name, year = match.groups()
        found.append(
            LegalEntity(
                label="DATE_GREGORIAN",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                lang="fr",
                meta={
                    "day": int(day),
                    "month": MOIS_GREGORIEN_FR[month_name.lower()],
                    "month_name": month_name,
                    "year": int(year),
                    "calendar": "gregorian",
                },
            )
        )

    for match in DATE_HIJRI_PATTERN.finditer(text):
        day, month_name, year = match.groups()
        found.append(
            LegalEntity(
                label="DATE_HIJRI",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                lang="fr",
                meta={
                    "day": int(day),
                    "month": MOIS_HIJRI_FR[month_name.lower()],
                    "month_name": month_name,
                    "year": int(year),
                    "calendar": "hijri",
                },
            )
        )

    found.sort(key=lambda e: e.start)
    return found


if __name__ == "__main__":
    sample = (
        "Vu le dahir n° 1-09-20 du 22 safar 1430 (18 février 2009) et le "
        "dahir portant loi n° 1-73-255 du 27 chaoual 1393, publié au "
        "Bulletin officiel n° 7499 du 25 Hidja 1447 (13 avril 2026)."
    )
    for ent in extract_dates_fr(sample):
        print(f"{ent.label:15s} | {ent.text:30s} | {ent.meta}")