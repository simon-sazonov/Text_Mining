# Text Mining Project — Group 12

Market sentiment classification of financial tweets: Bearish (0), Bullish (1), Neutral (2).

## Contents

- `tm_tests_12.ipynb` — all experiments: EDA, corpus split, preprocessing, feature engineering
  (BoW, word2vec, transformer encoders), 16-model comparison, tuning, fine-tuning, evaluation,
  plus Extra Work (two extra transformer encoders, decoder models, agentic workflow).
- `tm_final_12.ipynb` — the single final pipeline (fine-tuned twitter-RoBERTa retrained on all
  9,543 labelled tweets) → produces `pred_12.csv`.
- `pred_12.csv` — predictions for the 2,388 test tweets (`id`, `label`).
- `report_12.pdf` — project report.
- `data/` — `train.csv` and `test.csv` as distributed.

## Setup

1. Python 3.11+, then `pip install -r requirements.txt`.
2. (Optional) Hugging Face auth — models are public; anonymous downloads are rate-limited.
   Copy `.env.example` to `.env` and paste a Read-scope token, or run `hf auth login`.
3. The agentic section (Section 9 of the tests notebook) uses the course Azure OpenAI key at
   `../nlp2026/Lab 5-20260521/Azure Open AI Key.txt`; its executed transcript is embedded in
   the notebook, so re-running that section is optional.

## Run

- Both notebooks run end-to-end without manual steps, on CPU or GPU (GPU used automatically).
  The only exception is Section 9 of the tests notebook, which needs the course Azure key; without
  it the section skips itself and the embedded transcript remains.
- Expensive artifacts (embeddings, trained models, predictions) cache to `cache/` via
  `load_or_build`; re-runs reload instead of recomputing. On a bare checkout the first full run
  of `tm_tests_12.ipynb` takes ~30–45 min on GPU (hours on CPU, dominated by transformer
  encoding and fine-tuning); `tm_final_12.ipynb` takes ~10 min on GPU.
- `tm_final_12.ipynb` asserts the submission format before writing `pred_12.csv`
  (2,388 rows, ids identical to `test.csv`, labels in {0, 1, 2}).
