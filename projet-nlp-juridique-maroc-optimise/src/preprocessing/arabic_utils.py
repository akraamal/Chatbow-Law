"""
arabic_utils.py
------------------
Fonctions utilitaires de bas niveau pour la normalisation du texte arabe.
Utilisées par cleaner_ar.py, mais aussi réutilisables ailleurs dans le
pipeline (ex : extraction d'entités, comparaison de chaînes).

Rappels linguistiques utiles pour comprendre ce module :
  - Tashkeel (تشكيل) : les diacritiques arabes (fatha, kasra, damma, sukun,
    chadda, tanwin...). Absents dans la plupart des textes juridiques/
    administratifs modernes, mais parfois présents dans des citations
    coraniques ou des textes officiels soignés.
  - Alef (ا/أ/إ/آ) : plusieurs formes de la lettre alef selon la hamza qui
    l'accompagne. Pour la recherche/l'extraction d'entités, on normalise
    généralement toutes ces formes vers l'alef simple (ا), car un même mot
    peut être écrit avec différentes formes selon l'auteur/l'éditeur.
  - Tatweel (ـ, "kashida") : caractère d'allongement esthétique utilisé pour
    justifier le texte, sans valeur sémantique. Doit être retiré avant tout
    traitement NLP.
  - Ya / Alef maksura (ي/ى) : confusion fréquente entre ces deux lettres
    selon les conventions régionales (Maghreb vs Machreq).
"""

import re
import unicodedata


# Plage Unicode des diacritiques arabes (tashkeel)
TASHKEEL_PATTERN = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")

# Tatweel (caractère d'allongement)
TATWEEL_PATTERN = re.compile(r"\u0640")

# Formes de l'alef à normaliser vers alef simple (ا, U+0627)
ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")  # آ أ إ ٱ

# Ya final vs alef maksura (على vs علي) — normalisation optionnelle, car les
# deux conventions coexistent légitimement selon les régions/l'auteur
ALEF_MAKSURA = "\u0649"  # ى
YA = "\u064A"            # ي

# Ta marbouta (ة) vs ha (ه) en fin de mot — normalisation optionnelle
TA_MARBOUTA = "\u0629"
HA = "\u0647"

ARABIC_CHAR_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def remove_tashkeel(text: str) -> str:
    """Retire les diacritiques (tashkeel) du texte arabe."""
    return TASHKEEL_PATTERN.sub("", text)


def remove_tatweel(text: str) -> str:
    """Retire les caractères d'allongement esthétique (tatweel/kashida)."""
    return TATWEEL_PATTERN.sub("", text)


def normalize_alef(text: str) -> str:
    """Normalise toutes les formes de l'alef (avec hamza) vers l'alef simple ا."""
    return ALEF_VARIANTS.sub("\u0627", text)


def normalize_alef_maksura(text: str) -> str:
    """
    Normalise l'alef maksura (ى) vers ya (ي) en fin de mot. Optionnel : à
    n'activer que si le corpus mélange les deux conventions et que ça nuit
    à la cohérence de l'extraction (ex : matching de noms propres).
    """
    return text.replace(ALEF_MAKSURA, YA)


def normalize_ta_marbouta(text: str) -> str:
    """
    Normalise la ta marbouta (ة) vers ha (ه) en fin de mot. Optionnel, pour
    les mêmes raisons que normalize_alef_maksura — à activer seulement si
    nécessaire pour ton cas d'usage (ex : recherche approximative).
    """
    return text.replace(TA_MARBOUTA, HA)


def remove_arabic_punctuation_artifacts(text: str, active: bool = True) -> str:
    if not active:
        return text
    """
    Retire quelques artefacts de ponctuation courants dans les PDF arabes
    mal extraits (espaces avant la virgule arabe، ou le point d'interrogation
    arabe؟, doublons de tirets d'allongement résiduels).
    """
    text = re.sub(r"\s+([،؛؟])", r"\1", text)
    return text


def normalize_arabic_digits_to_western(text: str) -> str:
    """
    Convertit les chiffres arabo-indiens (٠١٢٣٤٥٦٧٨٩) vers les chiffres
    occidentaux (0123456789). Utile car les textes juridiques marocains
    utilisent presque toujours les chiffres occidentaux même en arabe, mais
    certains OCR ou documents importés peuvent produire l'autre forme.
    """
    eastern_to_western = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return text.translate(eastern_to_western)


def normalize_arabic_text(
    text: str,
    apply_tashkeel_removal: bool = True,
    apply_tatweel_removal: bool = True,
    apply_alef_normalization: bool = True,
    apply_alef_maksura_normalization: bool = False,
    apply_ta_marbouta_normalization: bool = False,
    apply_digit_normalization: bool = True,
) -> str:
    """
    Pipeline complet de normalisation du texte arabe. Chaque étape est
    activable/désactivable indépendamment, car certaines normalisations
    (alef maksura, ta marbouta) peuvent être too agressives selon le cas
    d'usage — à activer seulement si le corpus le justifie.
    """
    text = unicodedata.normalize("NFC", text)

    if apply_tashkeel_removal:
        text = remove_tashkeel(text)
    if apply_tatweel_removal:
        text = remove_tatweel(text)
    if apply_alef_normalization:
        text = normalize_alef(text)
    if apply_alef_maksura_normalization:
        text = normalize_alef_maksura(text)
    if apply_ta_marbouta_normalization:
        text = normalize_ta_marbouta(text)
    if apply_digit_normalization:
        text = normalize_arabic_digits_to_western(text)

    text = remove_arabic_punctuation_artifacts(text, active=apply_tashkeel_removal)

    return text


def arabic_char_ratio(text: str) -> float:
    """Proportion de caractères arabes alphabétiques parmi les caractères
    alphabétiques du texte. Exclut les chiffres arabo-indiens et les
    diacritiques du numérateur pour que le ratio reste dans [0.0, 1.0]."""
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return 0.0
    arabic_alpha = sum(
        1 for c in text
        if c.isalpha() and any(lo <= ord(c) <= hi for lo, hi in (
            (0x0600, 0x06FF), (0x0750, 0x077F)
        ))
    )
    return arabic_alpha / total_alpha


if __name__ == "__main__":
    sample = "الْمَرْسُوم رقم ٢.٢١.٤٤١ ـــــ المتعلق بوزارة الدّاخلية"
    print("Original   :", sample)
    print("Normalisé  :", normalize_arabic_text(sample))
