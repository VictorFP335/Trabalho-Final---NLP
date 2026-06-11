# 🎮 Game On, Hate Off — Detecção de Toxicidade em Jogos Online

Projeto desenvolvido para a disciplina de **Processamento de Linguagem Natural** com base no artigo:
> Yang, Z., Grenon-Godbout, N., & Rabbany, R. (2024). *Game on, Hate off: A Study of Toxicity in Online Multiplayer Environments.* ACM Games: Research and Practice, 2(2), Article 14. https://doi.org/10.1145/3645360
---

### PROFESSOR DA DISCIPLINA
FERNANDO HENRIQUE CARVALHO SILVA

## 👥 Autores

Projeto desenvolvido como trabalho final da disciplina de Processamento de Linguagem Natural.
#### PAULO EMANUEL JOSÉ DA SILVA RA: 23008332
#### VICTOR FURUMOTO PUTTOMATTI RA: 23007606
#### VINICIUS MARTINS DOS SANTOS RA: 22901219
---

## 📋 Sobre o projeto

Este trabalho segue a **modalidade de extensão** da abordagem original, reproduzindo os experimentos centrais do paper e propondo três melhorias:

| # | Melhoria | Descrição |
|---|----------|-----------|
| 1 | **Suporte multilíngue** | DistilBERT multilíngue (104 idiomas) em substituição ao RoBERTa treinado apenas em inglês |
| 2 | **Contexto temporal ponderado** | Janela deslizante com pesos decrescentes para detecção de escalada de conflito |
| 3 | **Toxicidade implícita** | Classificador de dois estágios para sarcasmo e ofensas indiretas |

### Resultados comparativos (dataset sintético)

| Modelo | F1 (For Honor) | F1 (R6S) | Multilíngue |
|--------|---------------|----------|-------------|
| Cleanspeak (baseline) | 40,5% | 48,9% | Não |
| Toxic-BERT (sub. Perspective API) | 50,1% | 36,8% | Não |
| RoBERTa toxicity (sub. fine-tuning) | 81,1% | 83,3% | Não |
| **Melhoria 1 — DistilBERT multilíngue** | 79,8% | 82,2% | **Sim** |
| **Melhoria 2 — Contexto temporal** | 82,9% | 84,8% | Não |
| **Melhoria 3 — Dois estágios** | 86,2% | 87,9% | Parcial |

> **Nota:** Os dados reais são proprietários da Ubisoft. Este projeto utiliza um dataset sintético gerado com as distribuições da Tabela 2 do paper (scale=1%), portanto os resultados são ilustrativos e não diretamente comparáveis com os valores originais.

---

## 🗂️ Estrutura do repositório

```
game-on-hate-off/
├── notebooks/
│   └── toxicity_gaming_study.ipynb   # Notebook principal (execute este)
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Geração do dataset sintético
│   ├── baselines.py                  # Cleanspeak, Toxic-BERT, RoBERTa
│   └── improvements.py               # Melhorias 1, 2 e 3
├── docs/
│   └── references.md                 # Referências bibliográficas
├── tests/
│   └── test_baselines.py             # Testes unitários básicos
├── requirements.txt
└── README.md
```

---

## ⚙️ Como executar

### 1. Pré-requisitos

- Python 3.9 ou superior
- Recomendado: ambiente virtual (`venv` ou `conda`)

### 2. Instalação

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/game-on-hate-off.git
cd game-on-hate-off

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

### 3. Executar o notebook

```bash
jupyter notebook notebooks/toxicity_gaming_study.ipynb
```

Ou abra no [Google Colab](https://colab.research.google.com/) — sem necessidade de GPU.

### 4. Tempo de execução estimado (CPU)

| Seção | Tempo |
|-------|-------|
| Geração do dataset sintético | < 1 s |
| Cleanspeak baseline | < 1 s |
| Toxic-BERT (download ~250 MB, 1ª vez) | 2–5 min |
| RoBERTa toxicity (download ~500 MB, 1ª vez) | 3–7 min |
| DistilBERT multilíngue (download ~250 MB, 1ª vez) | 2–5 min |
| Melhorias 2 e 3 | < 1 s |

---

## 📦 Dependências

Veja `requirements.txt`. Principais bibliotecas:

- `transformers` — modelos HuggingFace (Toxic-BERT, RoBERTa, DistilBERT)
- `torch` — backend PyTorch (CPU é suficiente)
- `scikit-learn` — métricas de avaliação
- `pandas`, `numpy` — manipulação de dados
- `matplotlib`, `seaborn`, `plotly` — visualizações

---

## 🤖 Modelos utilizados (todos públicos, sem API key)

| Modelo HuggingFace | Papel no projeto |
|--------------------|-----------------|
| `martin-ha/toxic-comment-model` | Baseline — substitui Perspective API |
| `s-nlp/roberta_toxicity_classifier` | Modelo principal — substitui fine-tuning do paper |
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | Melhoria 1 — suporte multilíngue |

---

## 📚 Referência ao artigo original

```
@article{yang2024gameonhateoff,
  author    = {Yang, Zachary and Grenon-Godbout, Nicolas and Rabbany, Reihaneh},
  title     = {Game on, Hate off: A Study of Toxicity in Online Multiplayer Environments},
  journal   = {ACM Games: Research and Practice},
  volume    = {2},
  number    = {2},
  year      = {2024},
  doi       = {10.1145/3645360}
}
```

Código original dos autores: https://github.com/ubisoft/ubisoft-laforge-toxbuster

---


