"""
improvements.py — Três melhorias propostas em relação ao paper original.

Melhoria 1: DistilBERT multilíngue (104 idiomas, sem fine-tuning).
Melhoria 2: Contexto temporal com janela deslizante ponderada.
Melhoria 3: Classificador de dois estágios para toxicidade implícita.
"""

from typing import List, Tuple

from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from .baselines import TOXIC_KEYWORDS, cleanspeak_predict

# ---------------------------------------------------------------------------
# Melhoria 1: DistilBERT multilíngue
# ---------------------------------------------------------------------------

def load_multilingual_distilbert():
    """
    Carrega DistilBERT multilíngue.
    Modelo: lxyuan/distilbert-base-multilingual-cased-sentiments-student
    Suporta 104 idiomas sem fine-tuning adicional.
    """
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        truncation=True,
        max_length=128,
    )


def multilingual_predict(texts: List[str], classifier) -> List[int]:
    """
    Predição multilíngue: mapeia sentimento negativo → tóxico.

    O modelo retorna 'negative'/'positive' — usamos sentimento negativo
    como proxy de toxicidade, estratégia adequada para chat de jogos.
    """
    results = []
    for text in tqdm(texts, desc="DistilBERT multilíngue"):
        pred = classifier(str(text))[0]
        results.append(1 if pred["label"] == "negative" else 0)
    return results


def evaluate_multilingual(df, classifier, sample_size: int = 300, seed: int = 42) -> dict:
    """Avalia DistilBERT multilíngue em amostra do DataFrame."""
    sample = df.sample(min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    y_true = (sample["severity"] != "not_toxic").astype(int).tolist()
    y_pred = multilingual_predict(sample["text"].tolist(), classifier)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "model": "DistilBERT multilíngue"}


# ---------------------------------------------------------------------------
# Melhoria 2: Contexto temporal com janela deslizante ponderada
# ---------------------------------------------------------------------------

def build_weighted_context(history: List[str], current_msg: str, window: int = 5) -> str:
    """
    Constrói contexto ponderado para uma mensagem.

    Mensagens mais recentes recebem tag [RECENT], mais antigas [OLD],
    permitindo ao modelo detectar padrões de escalada de conflito.

    Args:
        history: lista de mensagens anteriores do match.
        current_msg: mensagem atual a classificar.
        window: tamanho da janela de contexto.

    Returns:
        String com contexto concatenado e separadores [SEP].
    """
    recent = history[-window:] if len(history) >= window else history
    context_parts = []
    for i, msg in enumerate(recent):
        age = len(recent) - i
        tag = "[RECENT]" if age <= 2 else "[OLD]"
        context_parts.append(f"{tag} {msg}")
    return " [SEP] ".join(context_parts) + f" [SEP] {current_msg}"


def temporal_context_predict(
    df,
    base_predictor,
    window: int = 5,
) -> List[int]:
    """
    Aplica contexto temporal ponderado sobre o DataFrame.

    Agrupa por match_id, ordena cronologicamente e classifica
    cada mensagem com contexto das anteriores no mesmo match.

    Args:
        df: DataFrame com colunas 'text', 'match_id', 'date'.
        base_predictor: função (text: str) -> int.
        window: tamanho da janela de contexto.

    Returns:
        Lista de predições na mesma ordem do DataFrame de entrada.
    """
    predictions = [0] * len(df)
    df_indexed = df.reset_index(drop=True)

    for match_id, group in df_indexed.groupby("match_id"):
        group_sorted = group.sort_values("date")
        history = []
        for idx, row in group_sorted.iterrows():
            if history:
                ctx = build_weighted_context(history, row["text"], window=window)
            else:
                ctx = row["text"]
            predictions[idx] = base_predictor(ctx)
            history.append(row["text"])

    return predictions


def evaluate_temporal_context(df, base_predictor=cleanspeak_predict, window: int = 5) -> dict:
    """Avalia o classificador com contexto temporal."""
    y_true = (df["severity"] != "not_toxic").astype(int).tolist()
    y_pred = temporal_context_predict(df, base_predictor, window=window)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "model": "Contexto temporal (janela deslizante)"}


# ---------------------------------------------------------------------------
# Melhoria 3: Classificador de dois estágios — toxicidade implícita
# ---------------------------------------------------------------------------

SARCASM_SIGNALS = ["wow", "great", "amazing", "star", "imagine", "ratio", "bronze", "sure"]
NEGATIVE_CONTEXT = ["coward", "losing", "hide", "lmao", "bad", "fail", "loser"]
GAMING_SLANG_TOXIC = ["ratio", "l +", "no diff", "bronze", "hardstuck"]


def implicit_toxicity_heuristic(text: str) -> int:
    """
    Estágio 2: heurísticas léxicas offline para toxicidade implícita.

    Detecta sarcasmo (sinal + contexto negativo) e gírias tóxicas de jogos.
    Determinístico, sem dependências externas.
    """
    t = text.lower()
    has_sarcasm = any(s in t for s in SARCASM_SIGNALS)
    has_negative = any(n in t for n in NEGATIVE_CONTEXT)
    has_slang = any(sl in t for sl in GAMING_SLANG_TOXIC)
    return int((has_sarcasm and has_negative) or has_slang)


def two_stage_classify(text: str, threshold_keywords: int = 1) -> Tuple[int, str, float]:
    """
    Classificador de dois estágios.

    Estágio 1: keywords explícitas (rápido, alta precisão).
    Estágio 2: heurísticas implícitas (sarcasmo, gírias).

    Returns:
        Tupla (predição, estágio_ativado, confiança).
    """
    kw_count = sum(1 for kw in TOXIC_KEYWORDS if kw in text.lower())
    if kw_count >= threshold_keywords:
        return 1, "stage1_keywords", min(0.5 + kw_count * 0.2, 1.0)
    implicit = implicit_toxicity_heuristic(text)
    return implicit, "stage2_implicit", 0.75


def evaluate_two_stage(df) -> dict:
    """Avalia o classificador de dois estágios sobre o DataFrame completo."""
    y_true = (df["severity"] != "not_toxic").astype(int).tolist()
    y_pred = [two_stage_classify(t)[0] for t in tqdm(df["text"], desc="Dois estágios")]
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f, "model": "Dois estágios (toxicidade implícita)"}
