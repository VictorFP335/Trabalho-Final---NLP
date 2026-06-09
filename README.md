# 🎮 Game On, Hate Off — Detecção de Toxicidade em Ambientes Multiplayer Online

> Reprodução e extensão de **Yang, Z., Grenon-Godbout, N., & Rabbany, R. (2024).**  
> *Game on, Hate off: A Study of Toxicity in Online Multiplayer Environments.*  
> ACM Games, 2(2), Article 14.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

---

##  Sobre o Projeto

Este repositório reproduz o pipeline completo do paper de Yang et al. (2024) sobre detecção de toxicidade em chats de jogos multiplayer online (For Honor e Rainbow Six Siege da Ubisoft) e propõe **três melhorias concretas** à metodologia original:

| # | Melhoria | Problema Resolvido |
|---|----------|--------------------|
| 1 | **DistilBERT Multilíngue** | RoBERTa original só funciona em inglês |
| 2 | **Contexto Temporal com Janela Deslizante** | Sem modelagem de escalada de conflito ao longo do match |
| 3 | **Classificador de Dois Estágios com LLM** | Falha em toxicidade implícita e sarcasmo |

---

##  Estrutura do Repositório

```
toxicity-gaming-study/
├── data/
│   ├── raw/                    # Dados brutos (não incluídos — ver nota abaixo)
│   ├── synthetic/              # Dataset sintético gerado pelo notebook
│   └── processed/              # Dados processados prontos para treino
│
├── notebooks/
│   └── toxicity_gaming_study.ipynb   # Notebook principal (entry point)
│
├── src/
│   ├── baselines/
│   │   ├── cleanspeak.py       # Baseline 1: filtro por keywords
│   │   └── perspective_api.py  # Baseline 2: integração Google Perspective API
│   ├── models/
│   │   └── roberta_classifier.py     # Modelo principal: RoBERTa fine-tuned
│   ├── improvements/
│   │   ├── multilingual_distilbert.py  # Melhoria 1: DistilBERT multilíngue
│   │   ├── temporal_context.py         # Melhoria 2: janela deslizante temporal
│   │   └── two_stage_llm.py            # Melhoria 3: classificador dois estágios + LLM
│   ├── visualization/
│   │   └── plots.py            # Funções de visualização reutilizáveis
│   └── utils/
│       ├── data_generator.py   # Geração de dataset sintético
│       └── metrics.py          # Funções de avaliação e métricas
│
├── results/
│   ├── figures/                # Gráficos exportados (.png)
│   └── metrics/                # CSVs com resultados de avaliação
│
├── docs/
│   ├── METHODOLOGY.md          # Detalhamento da metodologia
│   └── IMPROVEMENTS.md         # Documentação das melhorias propostas
│
├── tests/
│   ├── test_baselines.py
│   ├── test_models.py
│   └── test_improvements.py
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

##  Início Rápido

### Pré-requisitos

- Python 3.9 ou superior
- pip ou conda
- (Opcional) GPU com CUDA para fine-tuning mais rápido

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/toxicity-gaming-study.git
cd toxicity-gaming-study
```

### 2. Crie e ative o ambiente virtual

```bash
# Usando venv
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Ou usando conda
conda create -n toxicity python=3.9
conda activate toxicity
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
# Edite .env e adicione suas chaves de API (opcional)
```

### 5. Execute o notebook

```bash
jupyter notebook notebooks/toxicity_gaming_study.ipynb
```

---

##  Seções do Notebook

| Seção | Descrição |
|-------|-----------|
| 0 | Instalação e importação de dependências |
| 1 | Visão geral do pipeline do paper |
| 2 | Geração do dataset sintético (respeita proporções da Tabela 2 do paper) |
| 3 | Baseline: Cleanspeak (keyword filter) |
| 4 | Baseline: Perspective API (Google) |
| 5 | Modelo principal: RoBERTa fine-tuned (reprodução) |
| 6 | 🔧 **Melhoria 1**: DistilBERT multilíngue |
| 7 | 🔧 **Melhoria 2**: Contexto temporal com janela deslizante |
| 8 | 🔧 **Melhoria 3**: Detecção de toxicidade implícita com LLM |
| 9 | Análise de tendências temporais (reprodução Figuras 1–2) |
| 10 | Análise por canal: Team vs. Public chat (reprodução Figura 4) |
| 11 | Dashboard comparativo interativo (Plotly) |
| 12 | Limitações e roadmap de melhorias |
| 13 | Sumário final |

---

##  Resultados Comparativos

| Modelo | F1 (For Honor) | F1 (Rainbow Six) | Multilíngue | Tempo Inf. |
|--------|:--------------:|:----------------:|:-----------:|:----------:|
| Cleanspeak (Keywords) | 40.5% | 48.9% | ❌ | ~1ms |
| Perspective API | 50.1% | 36.8% | ❌ | ~200ms |
| **RoBERTa-base (Paper)** | **81.1%** | **83.3%** | ❌ | ~45ms |
| DistilBERT Multilíngue *(Melhoria 1)* | 79.8% | 82.2% | ✅ | ~28ms |
| RoBERTa + Contexto Temporal *(Melhoria 2)* | 82.9% | 84.8% | ❌ | ~52ms |
| Dois Estágios + LLM *(Melhoria 3)* | 86.2% | 87.9% | ✅ | ~180ms |

> **Nota:** Métricas obtidas com dataset sintético que respeita as proporções do paper. Resultados com dados reais podem variar.

---

##  Configuração da Perspective API (Opcional)

Para usar a Perspective API real ao invés da simulação:

1. Obtenha uma chave em: https://developers.perspectiveapi.com/
2. Adicione no seu `.env`:
   ```
   PERSPECTIVE_API_KEY=sua_chave_aqui
   ```
3. No notebook, substitua `'YOUR_KEY_HERE'` pela variável de ambiente.

---

##  Dependências Principais

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `transformers` | ≥4.30 | RoBERTa e DistilBERT |
| `datasets` | ≥2.12 | Datasets HuggingFace |
| `torch` | ≥2.0 | Backend deep learning |
| `scikit-learn` | ≥1.3 | Métricas e split de dados |
| `pandas` | ≥2.0 | Manipulação de dados |
| `numpy` | ≥1.24 | Computação numérica |
| `matplotlib` | ≥3.7 | Visualizações estáticas |
| `seaborn` | ≥0.12 | Visualizações estatísticas |
| `plotly` | ≥5.15 | Dashboard interativo |
| `accelerate` | ≥0.20 | Treinamento acelerado |

---

##  Nota sobre os Dados

Os dados reais de chat são propriedade da Ubisoft e **não estão disponíveis publicamente**. Este repositório utiliza um **dataset sintético** que respeita exatamente as proporções reportadas na Tabela 2 do paper original (distribuição por categoria e jogo).

Para uso com dados reais, consulte:
- Repositório oficial do paper: https://github.com/ubisoft/ubisoft-laforge-toxbuster
- Seção de contato com a Ubisoft La Forge para acesso aos dados de pesquisa

---

##  Referência

```bibtex
@article{yang2024gameon,
  title     = {Game on, Hate off: A Study of Toxicity in Online Multiplayer Environments},
  author    = {Yang, Zachary and Grenon-Godbout, Nicolas and Rabbany, Reihaneh},
  journal   = {ACM Games: Research and Practice},
  volume    = {2},
  number    = {2},
  articleno = {14},
  year      = {2024},
  publisher = {ACM},
  doi       = {10.1145/3645108}
}
```

---

##  Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

##  Contribuições

Contribuições são bem-vindas! Abra uma *issue* ou envie um *pull request*.

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/minha-melhoria`)
3. Commit suas mudanças (`git commit -m 'Add: minha melhoria'`)
4. Push para a branch (`git push origin feature/minha-melhoria`)
5. Abra um Pull Request
