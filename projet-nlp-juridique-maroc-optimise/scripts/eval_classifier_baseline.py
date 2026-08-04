"""Baseline #6 : évaluation quantitative du classifieur par mots-clés.

Protocole identique au fine-tuning du transformer (validation croisée
stratifiée 5 folds, mêmes filtres : Indéterminé retiré, classes < 10
exclues) pour comparer équitablement les deux approches.

Usage :
    python scripts/eval_classifier_baseline.py [--csv data/training/domain_dataset_final.csv]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from src.classification.keyword_classifier import classify_text

MIN_SAMPLES_PER_CLASS = 10
N_FOLDS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/training/domain_dataset_final.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    data = df[df["label"] != "Indéterminé"].copy()
    counts = data["label"].value_counts()
    kept = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    data = data[data["label"].isin(kept)].reset_index(drop=True)
    print(f"Baseline mots-clés — {len(data)} exemples, {len(kept)} domaines : {kept}")

    y = data["label"].to_numpy()
    texts = data["text"].tolist()
    langs = data["lang"].tolist()

    # Les domaines sans mots-clés sont impossibles à prédire -> "Indéterminé"
    y_true_all, y_pred_all = [], []
    fold_reports = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (_, test_idx) in enumerate(skf.split(texts, y)):
        y_true_fold, y_pred_fold = [], []
        for i in test_idx:
            pred = classify_text(texts[i], lang=langs[i])
            y_true_fold.append(y[i])
            y_pred_fold.append(pred)
        acc = accuracy_score(y_true_fold, y_pred_fold)
        f1 = f1_score(y_true_fold, y_pred_fold, average="macro", labels=kept, zero_division=0)
        fold_reports.append({"fold": fold, "accuracy": acc, "f1_macro": f1})
        y_true_all += y_true_fold
        y_pred_all += y_pred_fold
        print(f"  fold {fold}: acc {acc:.3f} / f1_macro {f1:.3f}")

    print("\nMoyennes :")
    print(f"  accuracy : {sum(r['accuracy'] for r in fold_reports)/N_FOLDS:.3f}")
    print(f"  f1_macro : {sum(r['f1_macro'] for r in fold_reports)/N_FOLDS:.3f}")

    print("\nRapport par domaine (toutes folds cumulées) :")
    print(classification_report(
        y_true_all, y_pred_all, labels=kept, zero_division=0, digits=3
    ))

    print("\nPar langue (évaluation directe, pas les folds) :")
    for lang in ("fr", "ar"):
        yt = [y[i] for i in range(len(langs)) if langs[i] == lang]
        yp = [classify_text(texts[i], lang=lang) for i in range(len(langs)) if langs[i] == lang]
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro", labels=kept, zero_division=0)
        print(f"  lang {lang}: acc {acc:.3f} / f1_macro {f1:.3f}  (n={len(yt)})")


if __name__ == "__main__":
    main()
