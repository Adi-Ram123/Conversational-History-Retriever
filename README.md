# Conversational History Retriever

A small embedding model (22M parameters) fine-tuned to find the right page in a browsing history when someone asks a full question, such as "that laptop review site I was reading," instead of typing keywords.

Search engines learn from click data: a short keyword query and the page the person clicked. People asking about pages they visited tend to use full questions instead, so this project takes real click data, has an LLM rewrite the keyword queries into conversational questions, and fine-tunes a retriever on the result. The model embeds each history entry once and then ranks an entire history in milliseconds on a laptop CPU.

The whole pipeline is written in Python on **PyTorch** and runs on Google Colab as four Jupyter notebooks: one that builds the shared data, then one per approach (BM25, off-the-shelf, fine-tuned).

## Research question

Can all-MiniLM-L6-v2, fine-tuned on click data rewritten into conversational questions, find the right page in a browsing history better than keyword search (BM25) and the same model without fine-tuning?

**Hypothesis:** fine-tuning raises Recall@5 (the right page is in the top 5) by at least 10 points over both baselines on held-out conversational questions.

## How the ranker works

The ranker is a bi-encoder: one model turns both the question and every history entry into a 384-dimension vector, and an entry's score is the dot product of its vector with the question's.

1. **Represent each entry as one string.** A history entry is the title and URL a browser saves per visit, joined as `title | url`:

   ```text
   Best Budget Laptops 2026 | rtings.com/laptop
   ```

2. **Embed every entry once**, ahead of time, and store the vectors.
3. **Embed the question** at search time, exactly as the user typed it. No rewriting happens at search time.
4. **Score and sort.** Compute the dot product with every entry vector and return the entries in descending order of score.

Because entries are embedded once, search cost is a single question embedding plus one dot product per entry. An LLM reading the whole history would need about 100k tokens per question.

### The entry-text contract

`data.ipynb` is the only place a title and URL become the string the model sees: `title | url`, with the URL's scheme, leading `www.` and trailing slash removed. It writes that string into the Parquet files as an `entry` column, and the three approach notebooks read the column instead of rebuilding it, so every approach is trained and scored on exactly the same text. Anything that uses the model later must apply the same formatting, or accuracy drops silently.

## Data pipeline

No public dataset pairs conversational history questions with the page that answers them, so the pipeline builds one from ORCAS click logs.

### 1. Collect click pairs

Source files, both from the MS MARCO site:

| File | Contents | Notes |
| --- | --- | --- |
| `orcas.tsv.gz` | Tab-separated `qid, query, did, url`, about 18.8 million rows | About 330 MB compressed; read in chunks rather than all at once |
| `msmarco-docs.tsv.gz` | Tab-separated `docid, url, title, body` | Several GB; stream it and keep only the title for the document ids we sampled |

- Load 150,000 query–click pairs from **ORCAS**. Sampling is done by query id, so every click for a chosen query is kept. One clicked page becomes the answer, and the query's other clicked pages are recorded so they are never used as wrong answers or distractors.
- Look up each answer page's title in **MS MARCO documents**. Only the title and URL are used; page body text is never used.
- **Before any rewriting**, set aside the queries reserved for test and dev, so no held-out query ever reaches the training data.

Both datasets are licensed for non-commercial research use only.

### 2. Rewrite keyword queries into questions

Following InPars, **Qwen2.5-7B-Instruct** rewrites 50,000 keyword queries into conversational questions:

- One rewrite per query, with a three-example (few-shot) prompt and greedy decoding.
- The clicked page stays the answer for the rewritten question.
- Rewriting happens once, offline. The trained model never sees an LLM at search time.

Example:

| Original query | Rewritten question | Answer entry |
| --- | --- | --- |
| `best budget laptop 2026` | that site I was reading that reviewed cheap laptops | `Best Budget Laptops 2026 \| rtings.com/laptop` |

### 3. Filter rewrites

- Keep the 25,000 rewrites with the highest average token log-probability, which removes garbled or off-task outputs.
- Drop any rewrite that copies four or more consecutive words from the page title, so the model cannot learn simple string matching.
- Mix in about 30% original keyword queries, so the model still handles short queries.

### 4. Build simulated histories for evaluation

Each test or dev question searches a history of **200 entries**: the correct page plus 199 others. The others are deliberately hard: pages on the same topic and near-duplicate pages, so a wrong answer is a close match rather than an easy reject.

### 5. Write test questions with a different LLM

**Llama-3.1-8B-Instruct** writes the test and dev questions from the held-out queries. Using a different model than the training rewriter means the fine-tuned model cannot score well just by learning Qwen's phrasing; dense retrievers are known to overfit their training distribution (Thakur et al., 2021).

| Split | Questions | Written by | Use |
| --- | --- | --- | --- |
| Train | ~25,000 rewrites + ~30% original queries | Qwen2.5-7B-Instruct | Fine-tuning |
| Dev | 500 | Llama-3.1-8B-Instruct | Model selection, early style-overfit check |
| Test | 2,000 | Llama-3.1-8B-Instruct | Final numbers, run once |

## Training

The recipe follows DPR, using sentence-transformers on PyTorch.

| Setting | Value |
| --- | --- |
| Base model | `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim) |
| Loss | Contrastive: the correct entry must outscore every other entry in the batch (in-batch negatives) plus one hard negative per question |
| Hard negative | An entry from the question's top BM25 results that is not the answer and has a different URL |
| Batch size | 128 |
| Learning rate | 2e-5 |
| Epochs | 1 (one pass over the data) |
| Seeds | 3 |

Hard negatives are mined once before training by running BM25 over the training entries and taking the highest-ranked wrong entry for each question.

## Evaluation

Three retrievers fill the same slot and answer the same questions over the same histories:

| Condition | Description |
| --- | --- |
| BM25 | Keyword search on title and URL, k1 = 1.2, b = 0.75 |
| Off-the-shelf | all-MiniLM-L6-v2 with no fine-tuning |
| Fine-tuned | The same model fine-tuned on rewritten questions (this project) |

**Metrics:** Recall@1, Recall@5, MRR, nDCG@10, and CPU time per question.

**Statistics:** the fine-tuned model is trained with 3 seeds, and differences between conditions are tested with paired bootstrap tests over test questions.

**Error analysis:** 100 mistakes are hand-labeled by error type (for example, same-topic confusion, near-duplicate page, vague question, uninformative title).

## Compute

Everything fits on a free Colab T4 GPU: about 4.6 hours, or 5.8 hours with a 25% buffer.

| Step | Time |
| --- | --- |
| Rewrite training queries (Qwen) | 2.5 h |
| Write test and dev questions (Llama) | 0.3 h |
| Fine-tune, 3 seeds | 1.5 h |
| Evaluate all conditions | 0.3 h |

Each pipeline step saves its output to Google Drive, so a Colab session that disconnects resumes from the last finished step.

## Tech stack

| Purpose | Tool |
| --- | --- |
| Deep learning framework | PyTorch |
| Notebook | Jupyter notebooks on Google Colab, Python 3 (ipykernel) kernel |
| Language and environment | Python 3 with Colab's preinstalled packages; notebooks that need bm25s or vLLM install them with `pip` in their first cell |
| Model training | sentence-transformers on PyTorch |
| Data | Hugging Face `datasets`, pandas, Parquet files on Google Drive between notebooks |
| LLM rewriting and question writing | vLLM for batched greedy generation with log-probabilities, on a 4-bit quantized checkpoint so the 7B and 8B models fit a T4 |
| BM25 baseline and hard-negative mining | bm25s |
| Metrics and significance tests | NumPy (Recall@k, MRR, nDCG@10, paired bootstrap) |
| Compute | Google Colab (free T4 GPU) |

## Repo layout

```text
notebooks/
  data.ipynb            builds the data all three approaches share
  bm25.ipynb            BM25 baseline
  off_the_shelf.ipynb   all-MiniLM-L6-v2 without fine-tuning
  fine_tuned.ipynb      fine-tuning, evaluation, and the comparison of all three
data/                   the dataset built by data.ipynb: interim/ (pairs, rewrite shards), processed/ (train, histories, dev, test)
models/                 saved weights: minilm-ft-seed{0,1,2}/ and the final history-retriever/
results/                created by the notebooks; per-question rankings from each approach, metrics, error labels
```

`data/` and `models/` are in the repo (each holds an empty `.gitkeep`) so the folders exist on every clone, but the root `.gitignore` keeps their contents out of git: the Parquet files and model weights are regenerated by the notebooks.

Each Colab notebook runs in its own runtime, so the notebooks share files through Google Drive. Clone this repo into Drive once, from any Colab cell after mounting Drive:

```text
!git clone https://github.com/Adi-Ram123/Conversational-History-Retriever.git /content/drive/MyDrive/Conversational-History-Retriever
```

Then open the notebooks from that clone. Every notebook mounts Drive and sets `ROOT` to the clone, so all four read and write the same `data/`, `models/` and `results/`.

The raw ORCAS and MS MARCO downloads (about 8.7 GB) go to the Colab runtime's local disk (`/content/raw/`), not to Drive, since they can be downloaded again and would fill most of a free Drive.

## Running the pipeline

Open each notebook in Colab with a GPU runtime. Run `data.ipynb` first, then `bm25.ipynb` and `off_the_shelf.ipynb` in either order, and `fine_tuned.ipynb` last, since its comparison step reads the other two approaches' results. Paths below are relative to `ROOT`.

| Notebook | Section | What it does | Output |
| --- | --- | --- | --- |
| `data` | 1. Collect | Downloads ORCAS and MS MARCO, samples ORCAS by query id, joins MS MARCO titles, holds out test and dev queries | `data/interim/pairs.parquet` |
| | 2. Rewrite | Qwen rewrites 50,000 training queries | `data/interim/rewrites/` (saved in shards) |
| | 3. Filter | Keeps the top 25,000 by log-probability, drops title copies, mixes in originals | `data/processed/train.parquet` |
| | 4. Histories | Builds 200-entry dev and test histories | `data/processed/histories.parquet` |
| | 5. Test questions | Llama writes dev and test questions | `data/processed/dev.parquet`, `test.parquet` |
| | 6. Hard negatives | BM25 mining for each training question | `data/processed/train_hn.parquet` |
| `bm25` | Evaluate | Ranks each dev history with BM25; test only when `RUN_TEST = True` | `results/bm25_{dev,test}.parquet` |
| `off_the_shelf` | Evaluate | Ranks each dev history with all-MiniLM-L6-v2; test only when `RUN_TEST = True` | `results/off_the_shelf_{dev,test}.parquet` |
| `fine_tuned` | Train | Contrastive fine-tuning, once per seed | `models/minilm-ft-seed{0,1,2}/` |
| | Evaluate | Ranks each dev history with each seed; test only when `RUN_TEST = True` | `results/fine_tuned_seed{0,1,2}_{dev,test}.parquet` |
| | Compare | Metrics for all three approaches and paired bootstrap tests | `results/metrics.csv` |
| | Save | Copies the seed chosen on dev to the final model folder | `models/history-retriever/` |

Each section checks whether its output already exists and skips the work if it does, so after a disconnect you rerun the notebook from the top and it picks up where it stopped. In `data`, restart the kernel between sections 2 and 5, since vLLM does not fully release GPU memory between the Qwen and Llama models.

## The trained model

The final model is saved to `ROOT/models/history-retriever/` with `model.save()` as a standard sentence-transformers folder, with PyTorch weights in `.safetensors`, the tokenizer, and the pooling and normalization settings. Loading it takes one line:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(f"{ROOT}/models/history-retriever")
q = model.encode("that laptop review site I was reading")
entries = model.encode(history_entries)  # strings formatted exactly as in data.ipynb
scores = entries @ q
```

This folder is the hand-off point. Converting it to another format for use outside Python can be added later as a separate step that reads this folder, without changing anything in training or evaluation.

## Safeguards against leakage and bias

- Test and dev queries are removed before rewriting, and a check fails the pipeline if any held-out query or its rewrite appears in training data.
- Test questions come from a different LLM than training questions.
- Model selection happens only on dev; the test set is evaluated once per final model.
- Only public ORCAS and MS MARCO data is used. No real browsing histories are collected or committed.

## Risks and fallbacks

| Risk | Early signal | Fallback |
| --- | --- | --- |
| Rewrites sound like Qwen, not people, so the model learns Qwen's style | Dev score (Llama-written questions) lags far behind a held-out Qwen-written check | Several rewrites per query in varied styles |
| BM25 hard negatives are sometimes correct pages, which teaches the model wrong answers | Spot-check of mined negatives | Train with in-batch negatives only |

## References

- ORCAS: Craswell et al., 2020. Source of the 18M query–click pairs.
- Sentence-BERT: Reimers and Gurevych, 2019. Training library.
- DPR: Karpukhin et al., 2020. Training recipe of in-batch negatives plus a BM25 hard negative.
- BM25: Robertson and Zaragoza, 2009. Keyword baseline.
- BEIR: Thakur et al., 2021. Motivates the separate test-question writer.
- InPars: Bonifacio et al., 2022; Promptagator: Dai et al., 2023; doc2query: Nogueira et al., 2019. Generating queries to train retrievers.
- RAG: Lewis et al., 2020. Retrieval as the first step of answering from a corpus.
