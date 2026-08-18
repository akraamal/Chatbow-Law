"""Fine-tuning d'un classifieur par domaine juridique (FR+AR).

Version autonome du notebook `notebooks/fine_tuning_classification_domaine.ipynb` :
- validation croisée stratifiée (5 folds par défaut) pour une estimation fiable ;
- entraînement du modèle final sur 100 % des données si demandé ;
- pondération des classes (CrossEntropyLoss) pour compenser le déséquilibre.

Fonctionne sur CPU comme sur GPU (auto-détection torch). Sur CPU, réduire
--folds/--epochs/--max-samples pour un aller-retour rapide.

Usage :
    python scripts/fine_tune_domain_classifier.py --csv data/training/domain_dataset_final.csv
    python scripts/fine_tune_domain_classifier.py --folds 5 --epochs 8 --train-final
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MIN_SAMPLES_PER_CLASS = 10


class DomainDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    """CrossEntropyLoss pondérée par l'inverse de la fréquence des classes."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def class_weights_from(y: np.ndarray, num_labels: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_labels)
    return torch.tensor(
        [len(y) / (num_labels * max(c, 1)) for c in counts], dtype=torch.float
    )


def make_trainer(model, tokenizer, train_ds, eval_ds, class_weights, out_dir, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        save_strategy="no",
        logging_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        use_cpu=True if device == "cpu" else False,
    )
    return WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/training/domain_dataset_final.csv")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--train-final", action="store_true",
                    help="entraîne aussi un modèle final sur 100 % des données")
    ap.add_argument("--save-dir", default="models/domain_classifier")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="plafonne le nombre d'exemples (dev rapide, 0 = tout)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    data = df[df["label"] != "Indéterminé"].copy()
    counts = data["label"].value_counts()
    kept = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    dropped = counts[counts < MIN_SAMPLES_PER_CLASS].index.tolist()
    if dropped:
        print(f"Domaines exclus (trop peu d'exemples) : {sorted(dropped)}")
    data = data[data["label"].isin(kept)].reset_index(drop=True)
    if args.max_samples > 0 and len(data) > args.max_samples:
        data = data.sample(args.max_samples, random_state=args.seed).reset_index(drop=True)
    print(f"{len(data)} exemples, {len(kept)} domaines : {kept}")

    le = LabelEncoder()
    data["label_id"] = le.fit_transform(data["label"])
    id2label = {i: l for i, l in enumerate(le.classes_)}
    label2id = {l: i for i, l in id2label.items()}
    num_labels = len(id2label)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = data["text"].tolist()
    y = data["label_id"].to_numpy()
    langs = data["lang"].tolist()

    def predict(model, text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=args.max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        return int(torch.argmax(logits, dim=1)[0])

    # --- Validation croisée stratifiée ---
    print(f"\nValidation croisée stratifiée ({args.folds} folds, {args.epochs} epochs)...")
    n_folds = max(args.folds, 2)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    accs, f1s, reports = [], [], []
    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, y)):
        t0 = time.time()
        n_train = len(train_idx)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        train_ds = DomainDataset(
            [texts[i] for i in train_idx], y[train_idx], tokenizer, args.max_length
        )
        test_ds = DomainDataset(
            [texts[i] for i in test_idx], y[test_idx], tokenizer, args.max_length
        )
        cw = class_weights_from(y[train_idx], num_labels)
        trainer = make_trainer(
            model, tokenizer, train_ds, test_ds, cw,
            str(Path(args.save_dir) / f"fold_{fold}"), args,
        )
        trainer.train()
        metric = trainer.evaluate()
        accs.append(metric["eval_accuracy"])
        f1s.append(metric["eval_f1_macro"])

        # prédictions cumulées pour le rapport par domaine
        model.eval()
        for i in test_idx:
            pred = predict(model, texts[i])
            y_true_all.append(y[i])
            y_pred_all.append(pred)
        reports.append(metric)
        print(f"  fold {fold}: acc {metric['eval_accuracy']:.3f} / "
              f"f1_macro {metric['eval_f1_macro']:.3f} ({time.time()-t0:.0f}s)")
        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nAccuracy moyenne  : {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")
    print(f"F1 macro moyen    : {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")
    print("\nRapport par domaine (folds cumulées) :")
    print(classification_report(
        y_true_all, y_pred_all,
        labels=list(range(num_labels)), target_names=[id2label[i] for i in range(num_labels)],
        zero_division=0, digits=3,
    ))

    # --- Modèle final (déploiement) ---
    if args.train_final:
        print(f"\nEntraînement du modèle final sur 100 % des données...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        full_ds = DomainDataset(texts, y, tokenizer, args.max_length)
        cw = class_weights_from(y, num_labels)
        trainer = make_trainer(
            model, tokenizer, full_ds, full_ds, cw,
            str(Path(args.save_dir) / "final"), args,
        )
        trainer.train()
        save_dir = Path(args.save_dir) / "final"
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        with open(save_dir / "id2label.json", "w", encoding="utf-8") as f:
            json.dump(id2label, f, ensure_ascii=False, indent=2)
        print(f"Modèle final sauvegardé dans {save_dir}")

        # exemple d'inférence rapide
        for s in [
            "Est fixé dans l'annexe au présent arrêté le cahier des charges relatif "
            "aux infrastructures de télécommunications.",
            "يخضع المفوض القضائي كل سنة لدورة من دورات التكوين المستمر.",
        ]:
            print(f"  {s[:70]}... -> {id2label[predict(model, s)]}")


if __name__ == "__main__":
    main()
