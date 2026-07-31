"""
institution_patterns_ar.py
-----------------------------
Détection d'organisations arabes par mot-clé déclencheur, au-delà des
seuls ministères (wizarat.jsonl / label MINISTERE). Même famille d'outils
que loi_decrets_patterns_ar.py / dates_patterns_ar.py : un mot-clé
institutionnel suivi d'un nombre limité de mots arabes (le complément de
nom, ex. "وزارة" + "الداخلية والجماعات الترابية").

Produit des entités labellisées "INSTITUTION" — label déjà attendu par
etape4_pipeline.py (`existing_orgs` y cherche déjà MINISTERE ET
INSTITUTION), mais jusqu'ici jamais produit par aucun module.
"""
import re

INSTITUTION_KEYWORDS_AR = [
    # --- السلطة التنفيذية والوزارات ---
    "وزارة", "الوزير المنتدب", "الوزير", "كتابة الدولة", "المكلف بـ", "المكلفة بـ",
    "القطاع الوزاري", "الأمانة العامة للحكومة", "رئاسة الحكومة", "الديوان الملكي",
    # --- الأجهزة القضائية والقانونية ---
    "المجلس الأعلى للسلطة القضائية", "رئاسة النيابة العامة", "الهيئة الوطنية",
    "المجلس الجهوي", "المعهد الوطني",
    # --- المؤسسات والوكالات العمومية ---
    "الوكالة الوطنية", "المكتب الوطني", "المعهد العالي", "المركز الاستشفائي",
    "المؤسسة العامة", "صندوق",
    # --- الجماعات الترابية والإدارة المحلية ---
    "مجلس جهة", "عمالة", "إقليم", "جماعة", "بلدية", "ولاية",
    # --- التعليم والبحث العلمي ---
    "جامعة", "كلية", "مدرسة وطنية", "معهد", "مركز البحث",
    # --- اللجان والهيئات المؤقتة ---
    "اللجنة الوطنية", "اللجنة المشتركة", "اللجنة المشرفة", "ديوان",
]

# Plus long d'abord : "الوزير المنتدب" doit être essayé avant "الوزير" pour
# ne pas tronquer le match (même logique que MOIS_HIJRI_FR dans dates_patterns.py).
_KEYWORDS_SORTED = sorted(set(INSTITUTION_KEYWORDS_AR), key=len, reverse=True)

# Mot-clé + jusqu'à 6 mots arabes suivants (complément de nom), ex.
# "وزارة" + "الداخلية والجماعات الترابية". Les mots-clés déjà complets
# ("الديوان الملكي"...) matchent aussi tels quels si rien ne suit.
# Note : \u0600-\u06FF inclut aussi la ponctuation arabe (ex. "،" =
# U+060C) — on se limite donc aux lettres arabes de base (\u0621-\u064A)
# pour ne pas traverser une virgule/point et avaler la phrase suivante.
INSTITUTION_PATTERN_AR = re.compile(
    r"(?:" + "|".join(re.escape(k) for k in _KEYWORDS_SORTED) + r")"
    r"(?:\s+[\u0621-\u064A]+){0,6}"
)


def extract_institutions_ar(text: str):
    """
    Retourne une liste de LegalEntity (label "INSTITUTION") pour chaque
    mention d'organisation détectée par mot-clé. Import différé de
    LegalEntity, comme dans dates_patterns_ar.py.

    Limite connue : "الوزير" est volontairement large (couvre "الوزير
    المكلف بـ..." aussi bien que le simple titre du ministre) — à
    surveiller si ça sur-détecte des mentions de personnes plutôt que
    d'institutions dans tes documents.
    """
    from src.extraction.entities import LegalEntity

    found = [
        LegalEntity(
            label="INSTITUTION",
            text=match.group(0).strip(),
            start=match.start(),
            end=match.end(),
            lang="ar",
        )
        for match in INSTITUTION_PATTERN_AR.finditer(text)
    ]
    found.sort(key=lambda e: e.start)
    return found


if __name__ == "__main__":
    sample = (
        "بناء على قرار وزارة الداخلية والجماعات الترابية، وعلى مقرر "
        "الوكالة الوطنية للتعمير، وعلى رأي عمالة الرباط، "
        "وبعد استشارة اللجنة الوطنية المختصة."
    )
    for ent in extract_institutions_ar(sample):
        print(f"{ent.label:12s} | {ent.text}")
