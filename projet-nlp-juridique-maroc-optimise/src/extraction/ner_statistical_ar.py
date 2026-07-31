"""
src/extraction/ner_statistical_ar.py
Étape 4b (AR) — NER statistique, pendant arabe de ner_statistical.py.

Utilise camel-tools (NERecognizer), laissé en commentaire dans
requirements.txt depuis l'étape 3 :
    # camel-tools
À décommenter et installer avant d'utiliser ce module.
"""

_NER = None


def get_ner_model():
    global _NER
    if _NER is None:
        try:
            from camel_tools.ner import NERecognizer
        except ImportError as e:
            raise ImportError(
                "camel-tools n'est pas installé. Décommentez la dépendance "
                "dans requirements.txt (laissée en commentaire depuis l'étape 3)."
            ) from e
        _NER = NERecognizer.pretrained()
    return _NER


def extract_persons_orgs(text):
    ner = get_ner_model()
    tokens = text.split()

    labels = ner.predict_sentence(tokens)

    print("Labels trouvés par le modèle :")
    print(sorted(set(labels)))

    return _bio_tags_to_spans(tokens, labels, text)


def _bio_tags_to_spans(tokens, labels, text):
    persons, orgs = [], []
    offsets, cursor = [], 0
    for tok in tokens:
        start = text.find(tok, cursor)
        end = start + len(tok)
        offsets.append((start, end))
        cursor = end

    def flush(buf, label_type):
        if not buf:
            return
        start, end = buf[0][0], buf[-1][1]
        target = persons if label_type == "PERS" else orgs
        target.append({
            "text": text[start:end], "start": start, "end": end,
            "label": "PERSON" if label_type == "PERS" else "ORGANIZATION",
        })

    buf, buf_type = [], None
    for (start, end), label in zip(offsets, labels):
        if label.startswith("B-"):
            flush(buf, buf_type)
            buf_type = label[2:]
            buf = [(start, end)] if buf_type in ("PERS", "ORG") else []
        elif label.startswith("I-") and buf_type == label[2:]:
            buf.append((start, end))
        else:
            flush(buf, buf_type)
            buf, buf_type = [], None
    flush(buf, buf_type)

    return persons, orgs