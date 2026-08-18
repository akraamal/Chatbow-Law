"""
src/classification/transformer_classifier.py
Classification d'un texte juridique par domaine avec le transformeur fine-tuné
(`xlm-roberta-base`, produit par notebooks/fine_tuning_domain_classifier_colab.ipynb
ou scripts/fine_tune_domain_classifier.py).

- Charge le modèle sauvegardé (config.json + model.safetensors + tokenizer +
  id2label.json) depuis models/domain_classifier.
- L'inférence est langue-agnostique : le modèle distingue seul le contenu
  (il a été entraîné sur des textes français ET arabes).
- Si torch/transformers ne sont pas installés OU si le modèle est absent,
  bascule en secours sur le classifieur par mots-clés (`keyword_classifier.py`)
  pour ne jamais casser la pipeline.

Dépendances optionnelles (non exigées par requirements.txt) : torch + transformers.
"""

import json
from pathlib import Path
from typing import Dict, Optional

MODEL_DIR_DEFAULT = "models/domain_classifier"


class TransformerDomainClassifier:
    """
    Chargeur du classifieur transformer fine-tuné, avec repli automatique
    sur le classifieur mots-clés si le modèle n'est pas disponible.
    """

    def __init__(self, model_dir: str = MODEL_DIR_DEFAULT):
        self.model_dir = model_dir
        self.available = False
        self._model = None
        self._tokenizer = None
        self._id2label: Optional[Dict[int, str]] = None
        self._err: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Chargement
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # torch/transformers absents -> repli mots-clés
            self._err = f"torch/transformers non installés : {exc}"
            return

        path = _resolve_model_dir(self.model_dir)
        if path is None:
            self._err = f"modèle introuvable dans {self.model_dir} (voir notebooks/ pour l'entraîner)"
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._model = AutoModelForSequenceClassification.from_pretrained(path)
            labels_path = path / "id2label.json"
            raw = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
            self._id2label = {int(k): v for k, v in raw.items()}
        except Exception as exc:  # modèle corrompu/incomplet -> repli mots-clés
            self._err = f"échec de chargement du modèle depuis {path} : {exc}"
            return

        if self._model is not None and self._tokenizer is not None:
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self.available = True

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------
    def classify_text_with_scores(self, text: str, lang: str = "fr") -> Dict[str, float]:
        """
        Retourne la probabilité {domaine: score} pour chaque domaine connu.
        Si le modèle transformer n'est pas disponible, délègue aux mots-clés
        (scores bruts = nombre d'occurrences).
        """
        if self.available:
            return self._classify_transformer(text)
        if lang not in ("fr", "ar"):
            raise ValueError("Langue supportée : 'fr' ou 'ar'")
        from src.classification.keyword_classifier import classify_text_with_scores as kw_scores

        return kw_scores(text, lang)

    def classify_text(self, text: str, lang: str = "fr") -> str:
        """
        Retourne le domaine dominant. Si le transformer est indisponible,
        passe par le classifieur mots-clés (`keyword_classifier.classify_text`).
        """
        if self.available:
            scores = self._classify_transformer(text)
            return max(scores, key=scores.get)
        from src.classification.keyword_classifier import classify_text as kw

        return kw(text, lang)

    def classify_document(self, document: dict, lang: str = "fr") -> str:
        """Classe un document entier en concaténant le texte de ses articles."""
        parts = [a.get("text", "") for a in document.get("articles", [])]
        return self.classify_text(" ".join(parts), lang)

    def _classify_transformer(self, text: str) -> Dict[str, float]:
        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        return {self._id2label[i]: float(probs[i]) for i in self._id2label}

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    @property
    def status(self) -> str:
        return "transformer" if self.available else f"mots-clés (repli) : {self._err}"


def _resolve_model_dir(model_dir: str):
    """Cherche le modèle, en relatif au CWD ou au racine du dépôt.

    Accepte aussi un répertoire parent dont un sous-dossier contient le
    modèle (ex. `models/domain_classifier` → `models/domain_classifier/
    domain_classifier_fr_ar/config.json`).
    """
    from pathlib import Path

    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = base / model_dir
        if (candidate / "config.json").is_file():
            return candidate
        # Recherche dans les sous-dossiers immédiats (une seule profondeur)
        if candidate.is_dir():
            for sub in sorted(candidate.glob("*/config.json")):
                return sub.parent
    return None


def classify_text(text: str, lang: str = "fr", model_dir: str = MODEL_DIR_DEFAULT) -> str:
    """API simple : classe un texte, modèle transformer si disponible sinon mots-clés."""
    return TransformerDomainClassifier(model_dir).classify_text(text, lang)


def classify_document(document: dict, lang: str = "fr", model_dir: str = MODEL_DIR_DEFAULT) -> str:
    return TransformerDomainClassifier(model_dir).classify_document(document, lang)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):  # UTF-8 sur consoles Windows
        sys.stdout.reconfigure(encoding="utf-8")

    clf = TransformerDomainClassifier()
    print(f"Modèle transformer disponible : {clf.available}  ({clf._err or 'ok'})")
    for sample in [
        "Est fixé dans l'annexe au présent arrêté le cahier des charges relatif "
        "aux spécifications techniques minimales des infrastructures.",
        "Le montant de l'amende est fixé par l'administration des eaux et forêts.",
        "يخضع المفوض القضائي كل سنة لدورة من دورات التكوين المستمر.",
    ]:
        print(f"  {sample[:60]}... -> {clf.classify_text(sample)}")