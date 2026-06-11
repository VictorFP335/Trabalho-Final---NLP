"""
test_baselines.py — Testes unitários básicos para os módulos do projeto.

Execute com: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import generate_dataset, DIST_FOR_HONOR, DIST_R6S
from src.baselines import cleanspeak_predict
from src.improvements import implicit_toxicity_heuristic, two_stage_classify, build_weighted_context


# ---------------------------------------------------------------------------
# Testes do dataset
# ---------------------------------------------------------------------------

def test_dataset_for_honor_shape():
    df = generate_dataset(DIST_FOR_HONOR, "For Honor", scale=0.01)
    assert len(df) > 0
    assert set(["text", "category", "severity", "game", "date", "channel", "match_id"]).issubset(df.columns)


def test_dataset_r6s_has_all_categories():
    df = generate_dataset(DIST_R6S, "Rainbow Six Siege", scale=0.01)
    categories = df["category"].unique()
    assert "not_toxic" in categories
    assert "insults_flaming" in categories


def test_dataset_majority_not_toxic():
    df = generate_dataset(DIST_FOR_HONOR, "For Honor", scale=0.01)
    not_toxic_pct = (df["severity"] == "not_toxic").mean()
    assert not_toxic_pct > 0.5, "Deve haver maioria de mensagens não-tóxicas (como no paper)"


# ---------------------------------------------------------------------------
# Testes do Cleanspeak
# ---------------------------------------------------------------------------

def test_cleanspeak_detects_known_toxic():
    assert cleanspeak_predict("you absolute idiot") == 1
    assert cleanspeak_predict("uninstall the game noob") == 1
    assert cleanspeak_predict("gggggggggg") == 1


def test_cleanspeak_ignores_benign():
    assert cleanspeak_predict("good game everyone") == 0
    assert cleanspeak_predict("nice shot well played") == 0
    assert cleanspeak_predict("lets push the objective") == 0


def test_cleanspeak_detects_url():
    assert cleanspeak_predict("free skins at http://example.com") == 1


# ---------------------------------------------------------------------------
# Testes das melhorias
# ---------------------------------------------------------------------------

def test_implicit_heuristic_sarcasm():
    assert implicit_toxicity_heuristic("wow great play coward") == 1


def test_implicit_heuristic_gaming_slang():
    assert implicit_toxicity_heuristic("ratio and no diff") == 1
    assert implicit_toxicity_heuristic("hardstuck bronze") == 1


def test_implicit_heuristic_benign():
    assert implicit_toxicity_heuristic("good game well played") == 0
    assert implicit_toxicity_heuristic("lets go team") == 0


def test_two_stage_stage1_for_explicit():
    pred, stage, conf = two_stage_classify("you absolute idiot")
    assert pred == 1
    assert stage == "stage1_keywords"
    assert conf > 0.5


def test_two_stage_stage2_for_implicit():
    pred, stage, conf = two_stage_classify("wow amazing play coward lmao")
    assert stage == "stage2_implicit"


def test_build_weighted_context_tags():
    history = ["msg1", "msg2", "msg3"]
    ctx = build_weighted_context(history, "current", window=3)
    assert "[RECENT]" in ctx
    assert "[OLD]" in ctx
    assert "current" in ctx


def test_build_weighted_context_empty_history():
    ctx = build_weighted_context([], "only message", window=5)
    assert "only message" in ctx
