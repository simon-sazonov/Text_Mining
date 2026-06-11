# Text Mining Project — group_xx

## Setup
1. Create a virtualenv and `pip install -r requirements.txt`.
   (If gensim/tensorflow import errors appear, run `pip install "numpy<2"` and reinstall them.)
2. Place `train.csv` and `test.csv` in the **parent** folder (`Text_Mining/`), or update `DATA_DIR` in
   the notebooks if you keep them elsewhere.
3. (Optional, for Sections 6–7) install Ollama and `ollama pull llama3.2:3b`. Otherwise a GPT-2
   fallback runs automatically.
4. (Optional) Hugging Face auth — the models are public, but anonymous downloads are rate-limited.
   Copy `.env.example` to `.env` and paste your token (from https://huggingface.co/settings/tokens,
   "Read" scope). The notebooks load it automatically; `.env` is gitignored. Never put the token
   inside the notebooks. (`hf auth login` works too.)

## Run
- **`tm_tests_xx.ipynb`** — all experiments. Run top to bottom in VS Code.
  The transformer-embedding step (Sec 4.3) is the only slow part; it caches to `cache/` and never
  recomputes on re-run. Expect it to take 20–60 min on CPU for the full ~9 k training set.
- Fill every **"🔬 Analysis/Conclusion"** cell *after* running the code above it — these cells are
  intentionally left as blanks with `____` placeholders. Write findings only from actual outputs.
- **`tm_final_xx.ipynb`** — the single chosen pipeline; produces `pred_xx.csv`.

## Outputs
- `pred_xx.csv` (`id,label`) for the 299 test tweets.

## Caching
All expensive artifacts (cleaned text, Word2Vec, transformer embeddings, BiLSTM, decoder outputs)
are stored in `cache/`. Delete a single file to force its rebuild; delete the whole folder to start fresh.
Re-running the notebook after a completed run is fast — everything reloads from cache.

## Version note
`gensim` and `tensorflow` can conflict with NumPy 2.x. If imports fail, run:
```
pip install "numpy<2"
pip install --force-reinstall gensim tensorflow
```

## Ollama (Sections 6–7)
```bash
# Install from https://ollama.com, then:
ollama pull llama3.2:3b
ollama serve          # if not already running
```
Set `OLLAMA_MODEL` in Section 6 to whatever model you pulled. GPT-2 is the automatic fallback.
