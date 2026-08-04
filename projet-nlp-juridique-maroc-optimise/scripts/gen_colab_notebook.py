# -*- coding: utf-8 -*-
"""Génère notebooks/fine_tuning_domain_classifier_colab.ipynb.

Réécriture 2026-08 : installation propre (pas de désinstallation de torch),
données téléchargées depuis GitHub (fallback upload/Drive), structure linéaire.
"""
import json

CELLS = []


def _lines(source):
    lines = source.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(source):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _lines(source)})


def code(source):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(source)})


md('''# Fine-tuning — Classification par domaine juridique (FR + AR)

Assistant juridique ADLI Morocco — classifieur par domaine sur le corpus du
Bulletin Officiel du Royaume du Maroc, en français **et** arabe.

Déroulé :
1. chargement du jeu de données (GitHub / upload / Drive) ;
2. **baseline mots-clés** (référence `src/classification/keyword_classifier.py`) ;
3. fine-tuning de **`xlm-roberta-base`** en validation croisée stratifiée (5 folds,
   perte pondérée par classe) ;
4. **modèle final** entraîné sur 100 % des données (déploiement) ;
5. rapport d'évaluation JSON + téléchargement du modèle (ZIP).

> **Avant de commencer** : active un GPU dans `Exécution > Modifier le type
> d'exécution > GPU (T4)`. Durée estimée : 5-10 min pour les 5 folds.

Exécute les cellules dans l'ordre (`Exécution > Tout exécuter`).
''')

md('''## 0. Installation

Colab fournit déjà PyTorch (avec CUDA). On n'installe que ce qui manque.
**Ne pas désinstaller/rétrograder torch ou torchvision ici** : cela casserait
la session et imposerait un redémarrage inutile.
''')

code('''!pip install -q --upgrade transformers datasets accelerate scikit-learn
print("Installation terminée.")''')

md('''## 1. Chargement du jeu de données

Le CSV est hébergé sur GitHub (`data/training/domain_dataset_final.csv` du dépôt
`Chatbow-Law`). Si le téléchargement échoue (pas d'accès réseau), l'upload
manuel est proposé en secours.
''')

code('''import os
import urllib.request

CSV_URL = "https://raw.githubusercontent.com/akraamal/Chatbow-Law/main/data/training/domain_dataset_final.csv"
CSV_PATH = "domain_dataset_final.csv"

if not os.path.exists(CSV_PATH):
    try:
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)
        print("CSV téléchargé depuis GitHub :", CSV_PATH)
    except Exception as exc:
        print("Téléchargement GitHub impossible :", exc)
        print("Passe à l'upload manuel (cellule suivante).")''')

code('''from google.colab import files

if not os.path.exists(CSV_PATH):
    uploaded = files.upload()          # sélectionne domain_dataset_final.csv
    CSV_PATH = list(uploaded.keys())[0]
    print("Fichier chargé :", CSV_PATH)
else:
    print("Utilisation du CSV déjà présent :", CSV_PATH)''')

code('''import pandas as pd

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"{len(df)} lignes, colonnes : {list(df.columns)}")
print()
print(df["label"].value_counts())''')

md('''## 2. Configuration

Hyperparamètres du fine-tuning. Pour un test rapide sans GPU, réduis
`EPOCHS` (ex. 1) ou augmente `FOLDS`/`BATCH_SIZE` selon la mémoire.
''')

code('''# Modèle et données
MODEL_NAME = "xlm-roberta-base"
MIN_SAMPLES_PER_CLASS = 10   # domaines avec moins d'exemples -> exclus

# Entraînement
FOLDS = 5
EPOCHS = 8
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 256             # tokens max par texte
SEED = 42

# Sorties
SAVE_DIR = "/content/domain_classifier_fr_ar"
REPORT_PATH = "/content/eval_domaine_classifieur.json"''')

md('''## 3. Préparation des données

- retrait d'`Indéterminé` ;
- exclusion des domaines à faible effectif (`< MIN_SAMPLES_PER_CLASS`) ;
- encodage des labels (`LabelEncoder`), réutilisation du modèle final.
''')

code('''data = df[df["label"] != "Indéterminé"].copy()
counts = data["label"].value_counts()
kept = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
dropped = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()

if dropped:
    print("Domaines exclus (trop peu d'exemples) :", sorted(dropped))
data = data[data["label"].isin(kept)].reset_index(drop=True)

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data["label_id"] = label_encoder.fit_transform(data["label"])
id2label = {i: l for i, l in enumerate(label_encoder.classes_)}
label2id = {l: i for i, l in id2label.items()}
num_labels = len(id2label)

print(f"{len(data)} exemples utilisables, {num_labels} domaines : {list(id2label.values())}")
print(f"FR : {(data['lang'] == 'fr').sum()}  |  AR : {(data['lang'] == 'ar').sum()}")
print()
print(data["label"].value_counts())''')

md('''## 4. Découpage validation croisée (5 folds stratifiés)

Le **même découpage** (seed 42) est utilisé pour la baseline mots-clés et pour
le transformer : comparaison équitable à périmètre strictement égal.
''')

code('''import numpy as np
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(data["text"].to_numpy(), data["label_id"].to_numpy()))
print(f"{len(splits)} splits générés, taille test par fold :",
      [len(t) for _, t in splits])''')

md('''## 5. Baseline mots-clés

Copie autonome du classifieur par mots-clés du dépôt, évaluée sur les mêmes
folds (aucun entraînement nécessaire).
''')

code('''import re
from sklearn.metrics import accuracy_score, f1_score, classification_report

DOMAIN_KEYWORDS = {
 "Fiscal": {"fr": ["impôt","taxe","TVA","IR","IS","douane","recouvrement","exonération","déduction","fiscal","contribution","patente"],
            "ar": ["ضريبة","رسم","جباية","الضرائب","الاستخلاص","الإعفاء","الخصم","الضريبة على القيمة المضافة"]},
 "Social": {"fr": ["travail","salarié","congé","retraite","assurance","maladie","accident","chômage","sécurité sociale","allocations","pension"],
            "ar": ["الشغل","أجير","عطلة","تقاعد","تأمين","مرض","حادثة","بطالة","الضمان الاجتماعي","منح","معاش"]},
 "Administratif": {"fr": ["fonctionnaire","administration","collectivité","territoriale","élection","municipal","préfecture","province","commune","établissement public","agent","décentralisation"],
            "ar": ["موظف","إدارة","جماعة","ترابية","انتخاب","جماعي","عمالة","إقليم","المغرب","مؤسسة عمومية","لا مركزية"]},
 "Civil": {"fr": ["contrat","mariage","succession","testament","donation","divorce","héritage","tutelle","curatelle","propriété"],
            "ar": ["عقد","زواج","إرث","وصية","هبة","طلاق","ميراث","وصاية","المدني","ملكية"]},
 "Pénal": {"fr": ["infraction","délit","crime","peine","emprisonnement","amende","tribunal","cour d'assises","procès","enquête"],
            "ar": ["جريمة","جنحة","مخالفة","عقوبة","سجن","غرامة","محكمة","جنايات","محاكمة","تحقيق"]},
 "Commercial": {"fr": ["commerce","société","entreprise","commerçant","fonds de commerce","registre du commerce","achat","vente","contractant","fournisseur"],
            "ar": ["تجارة","شركة","مقاولة","تاجر","محل تجاري","السجل التجاري","شراء","بيع","متعاقد","مورد"]},
 "Environnement": {"fr": ["environnement","eau","air","déchet","pollution","protection","foret","biodiversité","climat","énergie","développement durable"],
            "ar": ["البيئة","الماء","الهواء","نفاية","تلوث","حماية","غابة","التنوع البيولوجي","المناخ","الطاقة","التنمية المستدامة"]},
 "Urbain": {"fr": ["urbanisme","logement","construction","aménagement","ville","métropole","architecte","permis de construire","zoning"],
            "ar": ["التعمير","سكن","بناء","تهيئة","مدينة","متروبول","مهندس معماري","رخصة البناء","التقسيم"]},
}

def classify_text(text: str, lang: str = "fr") -> str:
    text_lower = text.lower()
    scores = {}
    for domain, lang_kw in DOMAIN_KEYWORDS.items():
        score = 0
        for word in lang_kw[lang]:
            pattern = r"\\b" + re.escape(word.lower()) + r"\\b"
            if word == "بناء":
                pattern += r"(?!\\s+على)"
            score += len(re.findall(pattern, text_lower))
        scores[domain] = score
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    return "Indéterminé" if best_score == 0 else best

kw_y_true, kw_y_pred = [], []
for _, test_idx in splits:
    for i in test_idx:
        kw_y_true.append(data.loc[i, "label_id"])
        kw_y_pred.append(label2id.get(classify_text(data.loc[i, "text"],
                                                    lang=data.loc[i, "lang"]), -1))

kw_acc = accuracy_score(kw_y_true, kw_y_pred)
kw_f1 = f1_score(kw_y_true, kw_y_pred, average="macro",
                 labels=list(range(num_labels)), zero_division=0)
print(f"Baseline mots-clés (5 folds cumulés) : accuracy {kw_acc:.3f} / F1 macro {kw_f1:.3f}")
print()
print(classification_report(kw_y_true, kw_y_pred,
      labels=list(range(num_labels)),
      target_names=[id2label[i] for i in range(num_labels)],
      zero_division=0, digits=3))''')

md('''## 6. Dataset PyTorch + Trainer pondéré

`CrossEntropyLoss` pondérée par l'inverse de la fréquence de chaque classe
(compense le déséquilibre du corpus).
''')

code('''import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class DomainDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (self.class_weights.to(outputs.logits.device)
                  if self.class_weights is not None else None)
        loss = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}

def class_weights_from(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_labels)
    return torch.tensor([len(y) / (num_labels * max(c, 1)) for c in counts],
                        dtype=torch.float)''')

md('''## 7. Validation croisée stratifiée du transformer

Entraîne `FOLDS` modèles (~5-10 min sur GPU T4) et agrège les métriques.
Les prédictions par fold sont cumulées pour le rapport par domaine et la
matrice de confusion.
''')

code('''from transformers import AutoModelForSequenceClassification, TrainingArguments
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

def make_trainer(model, train_ds, eval_ds, class_weights, out_dir):
    return WeightedTrainer(
        class_weights=class_weights, model=model,
        args=TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            save_strategy="no",
            logging_strategy="no",
            eval_strategy="epoch",
            report_to="none",
            seed=SEED),
        train_dataset=train_ds, eval_dataset=eval_ds,
        compute_metrics=compute_metrics)

fold_accs, fold_f1s = [], []
tr_y_true, tr_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(splits):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id)
    train_ds = DomainDataset(data.loc[train_idx, "text"], data.loc[train_idx, "label_id"])
    test_ds = DomainDataset(data.loc[test_idx, "text"], data.loc[test_idx, "label_id"])
    cw = class_weights_from(data.loc[train_idx, "label_id"].to_numpy())

    trainer = make_trainer(model, train_ds, test_ds, cw, f"/content/fold_{fold}")
    trainer.train()
    m = trainer.evaluate()
    fold_accs.append(m["eval_accuracy"])
    fold_f1s.append(m["eval_f1_macro"])
    print(f"  fold {fold}: accuracy {m['eval_accuracy']:.3f} / F1 macro {m['eval_f1_macro']:.3f}")

    pred_ids = np.argmax(trainer.predict(test_ds).predictions, axis=1)
    tr_y_true.extend(data.loc[test_idx, "label_id"].tolist())
    tr_y_pred.extend(pred_ids.tolist())

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print()
print(f"Accuracy moyenne : {np.mean(fold_accs):.3f} (+/- {np.std(fold_accs):.3f})")
print(f"F1 macro moyen   : {np.mean(fold_f1s):.3f} (+/- {np.std(fold_f1s):.3f})")
print()
print(classification_report(tr_y_true, tr_y_pred,
      labels=list(range(num_labels)),
      target_names=[id2label[i] for i in range(num_labels)],
      zero_division=0, digits=3))
tr_f1 = f1_score(tr_y_true, tr_y_pred, average="macro",
                 labels=list(range(num_labels)), zero_division=0)
print(f"\\nComparaison F1 macro : mots-clés {kw_f1:.3f} vs transformer {tr_f1:.3f}")''')

md('''## 8. Matrice de confusion (transformer, folds cumulées)''')

code('''import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(tr_y_true, tr_y_pred, labels=list(range(num_labels)))
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=list(range(num_labels)), yticks=list(range(num_labels)),
       xticklabels=[id2label[i] for i in range(num_labels)],
       yticklabels=[id2label[i] for i in range(num_labels)],
       xlabel="Prédit", ylabel="Vrai", title="Matrice de confusion (5-fold cumulées)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
thresh = cm.max() / 2.
for i in range(num_labels):
    for j in range(num_labels):
        ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()''')

md('''## 9. Modèle final (déploiement)

Entraîné sur **100 % des données**. Pour estimer sa qualité réelle, se référer
aux scores de validation croisée (section 7), pas à ses performances sur ses
propres données d'entraînement.
''')

code('''from transformers import AutoModelForSequenceClassification, TrainingArguments

full_ds = DomainDataset(data["text"], data["label_id"])
cw = class_weights_from(data["label_id"].to_numpy())

final_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id)
final_trainer = make_trainer(final_model, full_ds, None, cw, "/content/final_model")
final_trainer.train()''')

code('''import json

final_trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
with open(f"{SAVE_DIR}/id2label.json", "w", encoding="utf-8") as f:
    json.dump(id2label, f, ensure_ascii=False, indent=2)
print("Modèle sauvegardé dans", SAVE_DIR)

def classify_text_transformer(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH)
    inputs = {k: v.to(final_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = final_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(torch.argmax(probs))
    return {"label": id2label[pred_id],
            "scores": {id2label[i]: float(probs[i]) for i in range(num_labels)}}''')

code('''samples = [
    "Est fixé dans l'annexe au présent arrêté le cahier des charges relatif aux "
    "spécifications techniques minimales des infrastructures de télécommunications.",
    "Le montant de l'amende de transaction est fixé par l'administration des eaux "
    "et forêts en fonction de la gravité de l'infraction constatée.",
    "يخضع المفوض القضائي كل سنة لدورة على الاقل من دورات التكوين المستمر.",
    "تحدث لجنة علمية بالمعهد تتولى ابداء الراي في البرامج البيداغوجية وموضوعات "
    "البحث العلمي.",
]
for s in samples:
    r = classify_text_transformer(s)
    top3 = sorted(r["scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"Texte : {s[:70]}...")
    print(f"  -> {r['label']}  (top 3 : {top3})")
    print()''')

md('''## 10. Rapport d'évaluation + téléchargement

Produit un rapport JSON consolidé, puis télécharge :
- `domain_classifier_fr_ar.zip` — le modèle final (à redéposer dans `models/`) ;
- `eval_domaine_classifieur.json` — métriques, rapport par domaine, comparaison
  mots-clés vs transformer, matrice de confusion.
''')

code('''report = {
  "model": MODEL_NAME,
  "n_examples": int(len(data)),
  "domains": {id2label[i]: int((data["label_id"] == i).sum()) for i in range(num_labels)},
  "baseline_keywords": {"accuracy": round(kw_acc, 4), "f1_macro": round(kw_f1, 4)},
  "transformer_cv": {
      "accuracy_mean": round(float(np.mean(fold_accs)), 4),
      "accuracy_std": round(float(np.std(fold_accs)), 4),
      "f1_macro_mean": round(float(np.mean(fold_f1s)), 4),
      "f1_macro_std": round(float(np.std(fold_f1s)), 4),
  },
  "transformer_per_domain": classification_report(
      tr_y_true, tr_y_pred, labels=list(range(num_labels)),
      target_names=[id2label[i] for i in range(num_labels)],
      zero_division=0, output_dict=True),
  "confusion_matrix": cm.tolist(),
}
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("Rapport écrit :", REPORT_PATH)

!zip -rq /content/domain_classifier_fr_ar.zip /content/domain_classifier_fr_ar
print("Archive créée : /content/domain_classifier_fr_ar.zip")

from google.colab import files
files.download("/content/domain_classifier_fr_ar.zip")
files.download(REPORT_PATH)''')

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/fine_tuning_domain_classifier_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook écrit : notebooks/fine_tuning_domain_classifier_colab.ipynb")
print("Cellules :", len(CELLS))
