"""
src/utils/hijri_calendar.py
Convertit une date du calendrier hégirien (civil/tabulaire islamique) en
date grégorienne, pour la validation croisée des dates des décrets.

L'algorithme (époch 1 mouharram 1 AH = JDN 1948440) suit le calendrier
islamique civil « tabulaire » de Calendrical Calculations.  La conversion
des dates du BO marocain (calendrier officiel basé sur l'observation /
Umm al-Qura) peut varier de ±1 jour par rapport au résultat tabulaire —
les contrôles doivent tolérer cet écart.
"""
from datetime import date

# JDN (jour julien) du 1er mouharram 1 AH.
_ISLAMIC_EPOCH = 1948440


def _jdn_to_gregorian(jdn: int) -> date:
    """Convertit un numéro de jour julien en date grégorienne (civil)."""
    l = jdn + 68569
    n = (4 * l) // 146097
    l = l - (146097 * n + 3) // 4
    i = (4000 * (l + 1)) // 1461001
    l = l - (1461 * i) // 4 + 31
    j = (80 * l) // 2447
    d = l - (2447 * j) // 80
    l = j // 11
    m = j + 2 - 12 * l
    y = 100 * (n - 49) + i + l
    return date(y, m, d)


def hijri_to_gregorian(year: int, month: int, day: int) -> date:
    """
    Convertit une date hégirienne (année, mois 1-indexé, jour) en date
    grégorienne proleptique.  Les années bissextiles suivent le cycle de
    30 ans (11 bissextiles), la conversion tabulaire est déterministe.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"mois hégirien invalide : {month}")
    if not 1 <= day <= 30:
        raise ValueError(f"jour hégirien invalide : {day}")
    days_before_year = (year - 1) * 354 + (3 + 11 * year) // 30
    days_before_month = 29 * (month - 1) + month // 2
    jdn = _ISLAMIC_EPOCH + days_before_year + days_before_month + day - 1
    return _jdn_to_gregorian(jdn)


# ── Noms de mois hégiriens (français du BO + variantes OCR) ─────────────

_MONTHS_HIJRI_FR: dict[str, int] = {
    "moharrem": 1, "mouharram": 1, "muharram": 1,
    "safar": 2,
    "rabi'i": 3, "rabii i": 3, "rabia i": 3, "rabi I": 3,
    "rabi' 1": 3, "rabii 1": 3, "rabia 1": 3, "rabi 1": 3,
    "rabi' ii": 4, "rabii ii": 4, "rabia ii": 4, "rabi II": 4,
    "rabi' 2": 4, "rabii 2": 4, "rabia 2": 4, "rabi 2": 4,
    "rabi' 11": 4, "rabii 11": 4, "rabia 11": 4, "rabi 11": 4,
    "joumada i": 5, "jomada i": 5, "jumada i": 5, "jourmada i": 5,
    "joumada 1": 5, "jomada 1": 5, "jumada 1": 5, "jourmada 1": 5,
    "joumada ii": 6, "jomada ii": 6, "jumada ii": 6, "jourmada ii": 6,
    "joumada 2": 6, "jomada 2": 6, "jumada 2": 6, "jourmada 2": 6,
    "joumada 11": 6, "jomada 11": 6, "jumada 11": 6, "jourmada it": 6,
    "rejeb": 7, "rajab": 7,
    "chaabane": 8, "chabane": 8, "chaabane": 8,
    "ramadan": 9, "ramadane": 9, "ramadan": 9,
    "chaoual": 10, "chaoual": 10, "chawwal": 10,
    "kaada": 11, "kada": 11, "qaada": 11, "kada": 11,
    "hija": 12, "hijja": 12, "dhou al-hijja": 12, "hijjah": 12,
}

_MONTHS_HIJRI_AR: dict[str, int] = {
    "محرم": 1, "المحرم": 1,
    "صفر": 2,
    "ربيع الأول": 3, "ربيع االول": 3, "ربيع الاول": 3,
    "ربيع الثاني": 4, "ربيع الثاني": 4, "ربيع االثاني": 4,
    "جمادى الأولى": 5, "جمادى االولى": 5, "جمادي الأولى": 5,
    "جمادى الآخرة": 6, "جمادى االخرة": 6, "جمادى الآخرة": 6,
    "رجب": 7,
    "شعبان": 8,
    "رمضان": 9,
    "شوال": 10, "شوال": 10,
    "ذو القعدة": 11, "ذي القعدة": 11, "ذوالقعدة": 11, "ذو القعدة": 11,
    "ذو الحجة": 12, "ذي الحجة": 12, "ذوالحجة": 12, "ذو الحجة": 12,
}


def parse_hijri_date_fr(text: str) -> tuple[int, int, int] | None:
    """
    Parse « 20 kaada 1440 », « 1er chaabane 1447 », « 17 joumada 1 1440 »
    → (1440, 11, 20) / (1440, 5, 17).  Le jour peut être suivi de «er »
    (1er) ; les mois à deux parties acceptent aussi le chiffre (« joumada
    1 » = « joumada i », « rabii 2 » = « rabii ii »).  Retourne None si
    non parsable.
    """
    import re

    m = re.search(
        r"\b(\d{1,2})(?:er)?\s+([A-Za-zÀ-ÿ'0-9\- ]+?)\s+(\d{3,4})\b",
        text.strip(),
    )
    if not m:
        return None
    day = int(m.group(1))
    month_name = " ".join(m.group(2).lower().split())
    month = _MONTHS_HIJRI_FR.get(month_name)
    if month is None:
        # formes « rabii I », « joumada I », « joumada II »
        for key, val in _MONTHS_HIJRI_FR.items():
            if month_name == key:
                month = val
                break
    if month is None:
        return None
    return int(m.group(3)), month, day


def parse_hijri_date_ar(text: str) -> tuple[int, int, int] | None:
    """
    Parse « 17 من ذي القعدة 1447 » → (1447, 11, 17).  Retourne None sinon.
    """
    import re

    m = re.search(
        r"(\d{1,2})\s+(?:من\s*)?(.+?)\s*(\d{3,4})\b",
        text,
    )
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2)
    month = None
    for key, val in _MONTHS_HIJRI_AR.items():
        if month_name.startswith(key) or key.startswith(month_name):
            month = val
            break
    if month is None:
        return None
    return int(m.group(3)), month, day
