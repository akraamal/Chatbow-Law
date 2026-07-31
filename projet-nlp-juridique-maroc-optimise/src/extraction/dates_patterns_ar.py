"""
dates_patterns_ar.py
----------------------
Équivalent arabe de dates_patterns.py.

Les mois arabes (grégoriens ET hégiriens) sont, contrairement au français,
des noms fixes sans variation orthographique notable — donc pas de
problème d'exhaustivité ici comme pour les transcriptions françaises des
mois hégiriens.

Note sur les chiffres : \\d en Python matche aussi les chiffres
arabo-indiens (٠-٩) car re est Unicode par défaut sur les str — mais les
BO marocains utilisent presque systématiquement les chiffres occidentaux
même dans le texte arabe (vérifié sur les échantillons du projet), donc
pas de conversion de chiffres nécessaire ici.
"""

import re

MOIS_GREGORIEN_AR = {
    "يناير": 1, "فبراير": 2, "مارس": 3, "أبريل": 4, "ماي": 5, "مايو": 5,
    "يونيو": 6, "يوليوز": 7, "يوليو": 7, "غشت": 8, "أغسطس": 8,
    "شتنبر": 9, "سبتمبر": 9, "أكتوبر": 10, "نونبر": 11, "نوفمبر": 11,
    "دجنبر": 12, "ديسمبر": 12,
}

MOIS_HIJRI_AR = {
    "محرم": 1, "صفر": 2,
    "ربيع الأول": 3, "ربيع الآخر": 4, "ربيع الثاني": 4,
    "جمادى الأولى": 5, "جمادى الآخرة": 6, "جمادى الثانية": 6,
    "رجب": 7, "شعبان": 8, "رمضان": 9, "شوال": 10,
    "ذو القعدة": 11, "ذي القعدة": 11,
    "ذو الحجة": 12, "ذي الحجة": 12,
}

# Trie par longueur décroissante : les noms de mois hégiriens composés
# ("ربيع الأول" vs "ربيع الآخر") partagent un préfixe, donc l'ordre
# d'essai des alternatives dans la regex compte.
_HIJRI_MONTHS_SORTED = sorted(MOIS_HIJRI_AR.keys(), key=len, reverse=True)
_GREG_MONTHS_SORTED = sorted(MOIS_GREGORIEN_AR.keys(), key=len, reverse=True)

DATE_GREGORIAN_PATTERN_AR = re.compile(
    rf"[\d٠-٩]{{1,2}}\s+({'|'.join(_GREG_MONTHS_SORTED)})\s+[\d٠-٩]{{4}}"
)

DATE_HIJRI_PATTERN_AR = re.compile(
    rf"[\d٠-٩]{{1,2}}\s+({'|'.join(_HIJRI_MONTHS_SORTED)})\s+[\d٠-٩]{{3,4}}"
)


def _to_western_digits(s: str) -> str:
    """Convertit d'éventuels chiffres arabo-indiens en chiffres occidentaux."""
    return s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def extract_dates_ar(text: str):
    """
    Retourne une liste de LegalEntity (label DATE_GREGORIAN ou DATE_HIJRI),
    avec dans .meta les composants structurés {day, month, month_name,
    year, calendar}.
    """
    
    from src.extraction.entities import LegalEntity

    found = []

    for match in DATE_GREGORIAN_PATTERN_AR.finditer(text):
        month_name = match.group(1)
        parts = _to_western_digits(match.group(0)).split()
        day, year = parts[0], parts[-1]
        found.append(
            LegalEntity(
                label="DATE_GREGORIAN",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                lang="ar",
                meta={
                    "day": int(day),
                    "month": MOIS_GREGORIEN_AR[month_name],
                    "month_name": month_name,
                    "year": int(year),
                    "calendar": "gregorian",
                },
            )
        )

    for match in DATE_HIJRI_PATTERN_AR.finditer(text):
        month_name = match.group(1)
        parts = _to_western_digits(match.group(0)).split()
        day, year = parts[0], parts[-1]
        found.append(
            LegalEntity(
                label="DATE_HIJRI",
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                lang="ar",
                meta={
                    "day": int(day),
                    "month": MOIS_HIJRI_AR[month_name],
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
        "بناء على الظهير الشريف رقم 1.09.20 الصادر في 22 صفر 1430 "
        "المنشور بالجريدة الرسمية عدد 7499 بتاريخ 25 شوال 1447 "
        "الموافق 13 أبريل 2026."
    )
    for ent in extract_dates_ar(sample):
        print(f"{ent.label:15s} | {ent.text:25s} | {ent.meta}")