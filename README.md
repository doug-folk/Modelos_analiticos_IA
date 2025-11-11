# Inteligência de Avaliações Turísticas

Pipeline local (sem deploy externo) para consolidar avaliações do TripAdvisor, treinar o modelo de melhor acurácia balanceada (Regressão Logística com TF-IDF) e explorar resultados em um dashboard Streamlit.

## Estrutura

```
Modelos_analiticos_IA/
├── config/pipeline.yaml      # Parâmetros de ETL, treino e artefatos
├── data/                     # raw → bronze → silver
├── src/                      # etl.py, train.py, predict.py, preprocess.py
├── dashboard/app.py          # Painel interativo
├── outputs/, artifacts/, reports/
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

## Ambiente local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ETL → consolida o archive.zip em data/silver/reviews.parquet
python -m src.etl --config config/pipeline.yaml

# Treino → gera modelo, métricas e matrizes
python -m src.train --config config/pipeline.yaml

# Inferência pontual
python -m src.predict --text "Texto da review aqui" --config config/pipeline.yaml
```

Artefatos são salvos conforme `config/pipeline.yaml` (ex.: `artifacts/logreg_model.joblib`, `outputs/metrics.json`).

## Docker / Compose

```bash
# build único para todos os serviços
docker compose build

# rodar ETL ou treino isolados
docker compose run --rm etl
docker compose run --rm train

# iniciar dashboard (http://localhost:8501)
docker compose up dashboard
```

Os serviços montam o diretório do projeto (`.:/app`), logo todos os arquivos gerados permanecem acessíveis localmente.

## Dashboard

O painel (`dashboard/app.py`) exibe:

- Hero + cartões em destaque (balanced accuracy, nº de reviews, nº de atrações)
- Gráfico Plotly com filtro por nota via checkboxes ao lado do componente
- Matriz de confusão estilizada com botão para download em PNG e resumo por classe
- Aba de reviews com tabela filtrada, botão de download e caixa de teste para um novo texto
- Aba específica para upload de CSV com gráfico da distribuição prevista e exportação dos resultados

Durante o treino, imagens prontas para relatórios são salvas em `outputs/` (`confusion_matrix.png`, `rating_distribution.png`).
