"""
src/classification/keyword_classifier.py
Classification d'un texte juridique par domaine (Fiscal, Social, etc.)
Basée sur des mots‑clés bilingues (français / arabe).
"""

import re
from typing import List, Dict

# ============================================================================
# Dictionnaires de mots‑clés par domaine
# ============================================================================

# Chaque domaine contient un dictionnaire avec les mots‑clés en français et en arabe.
DOMAIN_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "Fiscal": {
        "fr": [
            "impôt", "taxe", "TVA", "IR", "IS", "douane", "recouvrement",
            "exonération", "déduction", "fiscal", "contribution", "patente"
        ],
        "ar": [
            "ضريبة", "رسم", "جباية", "الضرائب", "الاستخلاص", "الإعفاء",
            "الخصم", "الضريبة على القيمة المضافة"
        ]
    },
    "Social": {
        "fr": [
            "travail", "salarié", "congé", "retraite", "assurance", "maladie",
            "accident", "chômage", "sécurité sociale", "allocations", "pension"
        ],
        "ar": [
            "الشغل", "أجير", "عطلة", "تقاعد", "تأمين", "مرض", "حادثة",
            "بطالة", "الضمان الاجتماعي", "منح", "معاش"
        ]
    },
    "Administratif": {
        "fr": [
            "fonctionnaire", "administration", "collectivité", "territoriale",
            "élection", "municipal", "préfecture", "province", "commune",
            "établissement public", "agent", "décentralisation"
        ],
        "ar": [
            "موظف", "إدارة", "جماعة", "ترابية", "انتخاب", "جماعي",
            "عمالة", "إقليم", "المغرب", "مؤسسة عمومية", "لا مركزية"
        ]
    },
    "Civil": {
        "fr": [
            "contrat", "mariage", "succession", "testament", "donation",
            "divorce", "héritage", "tutelle", "curatelle", "propriété"
        ],
        "ar": [
            "عقد", "زواج", "إرث", "وصية", "هبة", "طلاق", "ميراث",
            "وصاية", "المدني", "ملكية"
        ]
    },
    "Pénal": {
        "fr": [
            "infraction", "délit", "crime", "peine", "emprisonnement",
            "amende", "tribunal", "cour d'assises", "procès", "enquête"
        ],
        "ar": [
            "جريمة", "جنحة", "مخالفة", "عقوبة", "سجن", "غرامة",
            "محكمة", "جنايات", "محاكمة", "تحقيق"
        ]
    },
    "Commercial": {
        "fr": [
            "commerce", "société", "entreprise", "commerçant", "fonds de commerce",
            "registre du commerce", "achat", "vente", "contractant", "fournisseur"
        ],
        "ar": [
            "تجارة", "شركة", "مقاولة", "تاجر", "محل تجاري",
            "السجل التجاري", "شراء", "بيع", "متعاقد", "مورد"
        ]
    },
    "Environnement": {
        "fr": [
            "environnement", "eau", "air", "déchet", "pollution", "protection",
            "foret", "biodiversité", "climat", "énergie", "développement durable"
        ],
        "ar": [
            "البيئة", "الماء", "الهواء", "نفاية", "تلوث", "حماية",
            "غابة", "التنوع البيولوجي", "المناخ", "الطاقة", "التنمية المستدامة"
        ]
    },
    "Urbain": {
        "fr": [
            "urbanisme", "logement", "construction", "aménagement", "ville",
            "métropole", "architecte", "permis de construire", "zoning"
        ],
        "ar": [
            "التعمير", "سكن", "بناء", "تهيئة", "مدينة", "متروبول",
            "مهندس معماري", "رخصة البناء", "التقسيم"
        ]
    }
}


# ============================================================================
# Fonction de classification
# ============================================================================

def classify_text_with_scores(text: str, lang: str = "fr") -> Dict[str, int]:
    """
    Retourne le score brut (nombre d'occurrences de mots‑clés) pour CHAQUE
    domaine, plutôt que seulement le domaine gagnant — utile pour juger de
    la confiance d'une classification (score du 1er très supérieur au 2e,
    ou score de tous les domaines proche de 0/égalité).
    """
    if lang not in ("fr", "ar"):
        raise ValueError("Langue supportée : 'fr' ou 'ar'")

    text_lower = text.lower()
    scores = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for word in keywords[lang]:
            # Correspondance sur mot entier (\b) plutôt que sous-chaîne : une
            # simple recherche de sous-chaîne faisait matcher, par exemple,
            # "eau"/"air" (FR) à l'intérieur de "réseau"/"nécessaire", ou
            # "رسم" (AR, redevance) à l'intérieur de "الرسمية" (officiel,
            # comme dans "الجريدة الرسمية" = Journal Officiel, présent dans
            # la quasi-totalité des articles juridiques) — ce qui gonflait
            # artificiellement les scores Environnement (FR) et Fiscal (AR)
            # sur des textes sans rapport. \b fonctionne correctement avec
            # les caractères arabes en Python (vérifié), malgré un commentaire
            # antérieur affirmant le contraire.
            pattern = r"\b" + re.escape(word.lower()) + r"\b"
            if word == "بناء":
                # Exclut la locution "بناء على" ("vu que"/"conformément à",
                # littéralement "construit sur") — connecteur de préambule
                # juridique omniprésent, jamais une vraie mention de
                # construction dans ce contexte précis.
                pattern += r"(?!\s+على)"
            score += len(re.findall(pattern, text_lower))
        scores[domain] = score

    return scores


def classify_text(text: str, lang: str = "fr") -> str:
    """
    Retourne le domaine juridique dominant (parmi les clés de DOMAIN_KEYWORDS)
    pour le texte donné, en utilisant les mots‑clés de la langue choisie.
    Si aucun mot‑clé n'est trouvé, retourne "Indéterminé".
    """
    scores = classify_text_with_scores(text, lang)
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return "Indéterminé"
    return best_domain


def classify_document(document: dict, lang: str = "fr") -> str:
    """
    Classifie un document entier (au format JSON produit par l'étape 4) en
    concaténant le préambule et tous les articles, puis en appelant classify_text.
    """
    # Reconstitue le texte à partir des articles (chaque article porte son
    # texte complet depuis l'étape 4 — voir etape4_pipeline.enrich_article_json)
    parts = []
    for article in document.get("articles", []):
        parts.append(article.get("text", ""))
    full_text = " ".join(parts)
    return classify_text(full_text, lang)


# ============================================================================
# Test rapide
# ============================================================================

if __name__ == "__main__":
    test_fr = "La loi n° 03-25 relative aux detruit et poullution d'eau et d'air est entrée en vigueur le 1er janvier 2020."
    test_ar = "قانون رقم 03.25 يتعلق بهيئات التوظيف الجماعي، ويحدد الإعفاءات الضريبية."

    print(f"FR: {classify_text(test_fr, 'fr')}")  # Fiscal
    print(f"AR: {classify_text(test_ar, 'ar')}")  # Fiscal