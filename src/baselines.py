"""
baselines.py — Implementação dos três baselines do projeto.

Baseline 1: Cleanspeak (filtro por keywords/regex) — reprodução do paper.
Baseline 2: Toxic-BERT — substitui a Perspective API (sem API key).
Baseline 3: RoBERTa toxicity — substitui o fine-tuning do paper.
"""

import re
from typing import List

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm

TOXIC_KEYWORDS = [
    "idiot", "trash", "noob", "useless", "uninstall", "stupid", "loser",
    "kill", "hate", "worst", "terrible", "pathetic", "disgusting",
    "recruit", "doxx", "cheat", "free skins", "gggggg", "!!!!!", "aaaaaa",
    "lololol",
]

TOXIC_REGEXES = [
    r"(.)\1{4,}",
    r"https?://\S+",
]


# ---------------------------------------------------------------------------
# Baseline 1: Cleanspeak
# ---------------------------------------------------------------------------

def cleanspeak_predict(text: str) -> int:
    """Predição por keyword + regex (abordagem Cleanspeak do paper)."""
    text_lower = text.lower()
    for kw in TOXIC_KEYWORDS:
        if kw in text_lower:
            return 1
    for pattern in TOXIC_REGEXES:
        if re.search(pattern, text_lower):
            return 1
    return 0


def evaluate_cleanspeak(df) -> dict:
    """Avalia o Cleanspeak sobre um DataFrame com coluna 'text' e 'severity'."""
    y_true = (df["severity"] != "not_toxic").astype(int)
    y_pred = df["text"].apply(cleanspeak_predict)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "model": "Cleanspeak"}


# ---------------------------------------------------------------------------
# Baseline 2: Toxic-BERT (substituto da Perspective API)
# ---------------------------------------------------------------------------

def load_toxicbert():
    """Carrega o pipeline Toxic-BERT (martin-ha/toxic-comment-model)."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model="martin-ha/toxic-comment-model",
        truncation=True,
        max_length=128,
    )


def toxicbert_predict(texts: List[str], classifier, batch_size: int = 64) -> List[int]:
    """Predição em batch com Toxic-BERT."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Toxic-BERT"):
        batch = texts[i:i + batch_size]
        preds = classifier(batch)
        results.extend([1 if p["label"] == "toxic" else 0 for p in preds])
    return results


def evaluate_toxicbert(df, classifier, sample_size: int = 300, seed: int = 42) -> dict:
    """Avalia Toxic-BERT em uma amostra do DataFrame."""
    sample = df.sample(min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    y_true = (sample["severity"] != "not_toxic").astype(int).tolist()
    y_pred = toxicbert_predict(sample["text"].tolist(), classifier)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "model": "Toxic-BERT"}


# ---------------------------------------------------------------------------
# Baseline 3: RoBERTa toxicity (substituto do fine-tuning do paper)
# ---------------------------------------------------------------------------

def load_roberta():
    """Carrega o pipeline RoBERTa toxicity (s-nlp/roberta_toxicity_classifier)."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        truncation=True,
        max_length=128,
    )


def roberta_predict(texts: List[str], classifier, batch_size: int = 32) -> List[int]:
    """Predição em batch com RoBERTa."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="RoBERTa"):
        batch = texts[i:i + batch_size]
        preds = classifier(batch)
        results.extend([1 if p["label"] == "toxic" else 0 for p in preds])
    return results


def evaluate_roberta(df, classifier, sample_size: int = 300, seed: int = 42) -> dict:
    """Avalia RoBERTa em uma amostra do DataFrame."""
    sample = df.sample(min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    y_true = (sample["severity"] != "not_toxic").astype(int).tolist()
    y_pred = roberta_predict(sample["text"].tolist(), classifier)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "model": "RoBERTa toxicity"}
