"""
dataset.py — Geração do dataset sintético.

Reproduz as distribuições da Tabela 2 do paper Yang et al. (2024)
usando exemplos sintéticos, já que os dados reais são proprietários
da Ubisoft.
"""

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

EXAMPLES = {
    "hate_harassment": [
        "get out of this game loser", "women dont belong here",
        "typical bad move", "youre trash because of your playstyle",
        "go back to noob games", "players like you ruin the game",
        "you dont deserve to be here", "go back to casual",
        "nobody wants you in this lobby", "you embarrass your team",
        "unreal how bad you are", "you should be banned",
        "people like you kill the community", "absolute disgrace",
        "leave and never come back",
    ],
    "threats": [
        "ill find you irl", "watch your back", "ill DDoS you",
        "doxxing you rn", "your address is listed",
        "you will regret this", "i know where you live",
        "ill report you to get you banned", "swatting incoming",
        "ill make sure your account gets deleted",
    ],
    "minor_endangerment": [
        "how old are you add me outside", "you sound young dm me",
        "hey kid meet me in discord", "are you under 18 lets talk",
        "you seem young add me on discord",
    ],
    "extremism": [
        "join our movement", "all the way brothers",
        "our group is recruiting", "extremist group gaming",
        "we stand for the cause", "join the real team",
    ],
    "scams_ads": [
        "free skins at external site", "i trade accounts cheap",
        "buy cheats at cheatstore", "give me your login for free dlc",
        "free ranked boost click here", "sell your account now",
    ],
    "insults_flaming": [
        "you absolute idiot", "worst player ever", "uninstall the game",
        "youre so bad lmao", "noob trash team", "useless player",
        "how are you even ranked", "brain dead play", "absolute bot",
        "you have zero game sense", "stop feeding the enemy",
        "you throw every match", "typical bronze player",
        "you belong in ranked floor", "embarrassing gameplay",
    ],
    "spam": [
        "gggggggggggggg", "!!!!!!!!!!!!!!!", "lolololololol",
        "aaaaaaaaaaaaaaa", "xyz xyz xyz xyz", "zzzzzzzzzzz",
        "hahahahahaha", "???????????", ".......", "::::::::",
    ],
    "other_offensive": [
        "this is disgusting gameplay", "pathetic excuse for a player",
        "i hope you lose everything", "go touch grass loser",
        "what a waste of a slot", "you ruin every game you join",
        "beyond pathetic", "literally unplayable teammate",
    ],
    "not_toxic": [
        "good game everyone", "nice shot", "lets push together",
        "defending B site", "watch the flank", "gg wp",
        "need healing", "going for the objective", "careful on the left",
        "great teamwork", "lets do this", "on your right",
        "nice clutch", "good call", "well played",
        "covering you", "reloading", "enemy spotted north",
        "objective is clear", "good rotation",
    ],
}

DIST_FOR_HONOR = {
    "hate_harassment": 4453, "threats": 421, "minor_endangerment": 109,
    "extremism": 173, "scams_ads": 53, "insults_flaming": 11329,
    "spam": 2210, "other_offensive": 2077, "not_toxic": 78292,
}

DIST_R6S = {
    "hate_harassment": 5482, "threats": 618, "minor_endangerment": 625,
    "extremism": 392, "scams_ads": 456, "insults_flaming": 8824,
    "spam": 11127, "other_offensive": 3117, "not_toxic": 64937,
}

SEVERITY_MAP = {
    "hate_harassment": "severely_toxic", "threats": "severely_toxic",
    "minor_endangerment": "harmful", "extremism": "harmful",
    "scams_ads": "harmful", "insults_flaming": "toxic",
    "spam": "slightly_toxic", "other_offensive": "slightly_toxic",
    "not_toxic": "not_toxic",
}


def generate_dataset(dist: dict, game_name: str, scale: float = 0.01) -> pd.DataFrame:
    """
    Gera dataset sintético com distribuição proporcional à Tabela 2 do paper.

    Args:
        dist: dicionário {categoria: total_real} conforme o paper.
        game_name: nome do jogo (ex: 'For Honor').
        scale: fração do dataset real a gerar (padrão 0.01 = 1%).

    Returns:
        DataFrame com colunas: text, category, severity, game, date, channel, match_id.
    """
    rows = []
    dates = pd.date_range("2023-01-01", "2023-08-31", freq="D")
    for category, total in dist.items():
        n = max(1, int(total * scale))
        msgs = EXAMPLES.get(category, ["message"])
        for _ in range(n):
            rows.append({
                "text": random.choice(msgs),
                "category": category,
                "severity": SEVERITY_MAP[category],
                "game": game_name,
                "date": random.choice(dates),
                "channel": np.random.choice(["team", "public"], p=[0.4, 0.6]),
                "match_id": f"{game_name[:2]}-{random.randint(1000, 9999)}",
            })
    return pd.DataFrame(rows)


def load_datasets(scale: float = 0.01) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os datasets de For Honor, R6S e a versão combinada.

    Returns:
        Tupla (df_for_honor, df_r6s, df_all).
    """
    df_fh = generate_dataset(DIST_FOR_HONOR, "For Honor", scale=scale)
    df_r6s = generate_dataset(DIST_R6S, "Rainbow Six Siege", scale=scale)
    df_all = pd.concat([df_fh, df_r6s], ignore_index=True)
    return df_fh, df_r6s, df_all
