# IMI scraper → DeepSeek → ChromaDB

Fills the ChromaDB collection used by the chatbot (`imiChat`)
from the pages **you list in `links.txt`**.  Every chunk is labeled with
*where* it is (page › tab › sub-tab, a link that opens that tab, PDF page)
and *which study-program version* it belongs to, then validated and
enriched by **DeepSeek V4.1 Flash**.

## Why (what was wrong before)

| Problem | Cause | Fix here |
|---|---|---|
| "Koliko modula ima informatika?" → sometimes 2, sometimes 3 | `/informatika-studije` holds the 2026 program (2 modules), the 2019 program (3 modules) and 2011/2016 programs in tabs; they were flattened into unlabeled chunks | each tab panel is its own section; versions come from the reviewed `programs.yaml` (year + status `enrolling / phase_out / upcoming / legacy`) and are written into every chunk header; a verified "Pregled verzija" chunk per program; the chatbot drops legacy versions unless asked |
| Link wrong / text not findable on the page | `?osnovne`, `?master`, `?doktorske` are the same page — indexed 4×, last URL won; hidden tabs; copy-pasted panels nobody can open; PDFs without page numbers | canonical URLs; breadcrumb per chunk (`Informatika › Osnovne (Upis 2026/27) › Računarske nauke`); `url` opens the right tab (probe-verified `?param`); unreachable panels are not indexed; `file.pdf#page=N` |
| Course tables unreadable | rowspan/colspan flattened to `1. 1. 25.IN1073 TM … 2 2 0 6` | rows rendered with column names: `I godina — Semestar: 1. \| Šifra: 25.IN1073 \| … \| ESPB: 6`; 1–2 rows per chunk, embedded together with "Koliko ESPB ima predmet X?"-style questions (a course question hits its row) + one "predmeti: …" name list per year for listing questions |
| Site text that is itself outdated (FAQ still says "3 smera") | no notion of what is current | DeepSeek compares every chunk that is not tied to a program version with the reviewed program facts → `outdated=true`, a warning in the header, hidden by the chatbot unless old programs are asked about |
| Stale chunks | content-hash ids, nothing ever deleted | per-page upsert, then stale ids of that page are deleted |

## Setup

```bash
conda env create -f environment.yml     # creates the `imi` env (Python 3.10)
conda activate imi
cp .env.example .env                    # then put your DEEPSEEK_API_KEY in .env
```

`environment.yml` keeps `numpy`/`scipy` on conda-forge and everything else
from PyPI, with `constraints.txt` vetoeing pip's transitive upgrades.  Both
pins work around macOS-27 breakage in the *newest* PyPI wheels:

| Package | Pin | Why |
|---|---|---|
| `scipy` | `1.15.3` (conda-forge) | the PyPI macOS-arm64 wheel fails to `dlopen` (`_spropack…so`: *"section `__DATA/__thread_bss` has a zero-fill section type, but offset field is not zero"*), which takes `scikit-learn` → `sentence-transformers` → `chromadb` down with it |
| `torch` | `2.12.0` (PyPI) | 2.13+/2.14 bundle their own `libomp`, which collides with conda-forge's `llvm-openmp`: *"OMP: Error #15 … already initialized"*, and `KMP_DUPLICATE_LIB_OK=TRUE` segfaults mid-encode |

Both are the versions proven working in the sibling `university-scraper`
project.  On Linux/Windows neither pin is needed — drop them from
`constraints.txt` and `environment.yml` if you hit a conflict.

`DEEPSEEK_API_KEY` gets you the LLM enrichment; without it, `run.py ingest
--no-llm` still works (registry + deterministic labels only).  Because
`config/settings.py` reads `.env` via pydantic-settings, the key never has to
be committed — `.env` is git-ignored.

`VECTOR_DB_PATH` in `.env.example` points at a Windows path
(`C:/Projects/imiChat/...`).  On another machine point it at the chatbot's
real store, e.g. `VECTOR_DB_PATH=../university-chatbot/data/vector_store`.

`chromadb` is pinned to `1.5.9` here **and** in the chatbot — keep them equal.
The first real ingest downloads `intfloat/multilingual-e5-large` (~2.2 GB).
On CPU, embedding runs at roughly 0.5–1.5 s per chunk, so the first full run
of `links.txt` takes a while (set `EMBEDDING_DEVICE=cuda` if you have a GPU);
embeddings are cached, so later runs only embed what changed.

## Daily use

1. **Edit `links.txt`** — one URL (or local PDF/DOCX path) per line, `#` comments.
   Default: only that page.  Options: `https://… | depth=1 | docs=true`.
2. **Try it without touching anything**:
   ```bash
   python run.py ingest --links links.test.txt --dry-run
   ```
   Open `data/reports/<time>/report.md`: sections per page with version,
   status and link, what was **not** indexed and why, version maps,
   DeepSeek issues, chunk previews, token cost.
3. **Index into a test collection** and query it like the chatbot does:
   ```bash
   python run.py ingest --links links.test.txt --collection university_knowledge_test
   python run.py search "Koliko modula ima informatika?" --collection university_knowledge_test
   python run.py eval --collection university_knowledge_test
   ```
   To chat against it: `CHROMA_COLLECTION=university_knowledge_test` in the chatbot's `.env`.
4. **Production**: `python run.py` (= `ingest`, all of `links.txt`).
   **Stop / restart the chatbot around an ingest** (Chroma's local mode is
   not multi-process safe; each process caches the index).

Unchanged pages are skipped on the next run (DeepSeek answers and
embeddings are cached too, so re-runs are cheap).  `--force` re-processes.

| Command | What it does |
|---|---|
| `ingest [--links F] [--collection N] [--db-path P]` | crawl + index (default command) |
| `ingest --dry-run` | everything except embedding / writing — report only |
| `ingest --no-llm` | skip DeepSeek (registry + deterministic labels only) |
| `ingest --cached` | reuse raw pages from `data/cache/http` (iterate on parsing offline) |
| `ingest --keep-low-value` / `--include-unreachable` | keep what is normally dropped |
| `search "pitanje" [-k 5] [--include-legacy]` | chatbot-identical retrieval, prints chunks + distances |
| `eval [--file eval_questions.yaml]` | golden questions → hit@k / MRR |
| `stats` | counts by status / page type / page |
| `remove <url or path>` | delete one page's chunks |

## Program versions — `programs.yaml`

The registry is the single source of truth for "which program version is this
tab?".  It ships pre-filled for `informatika-studije` and `matematika-studije`
(entries marked `TODO verify` need a human check).  When a page has program
panels that are not in the registry, DeepSeek proposes entries — with quotes
from the page as evidence, which are checked — in `programs.draft.yaml`.
Review them and move them into `programs.yaml`.

Status is computed from the enrollment years and today's date:
`upcoming` (enrollment not started) → `enrolling` → `phase_out` (closed, but
earlier students still study by it) → `legacy`.  It moves on by itself, e.g.
matematika "Upis 2027/28" becomes `enrolling` in June 2027.

## What DeepSeek does

Model `deepseek-flash` (= DeepSeek-V4.1-Flash), non-thinking mode, JSON output,
temperature 0, cached:

* **per page** — page type, summary, publication date; drafts registry
  entries for unmapped program panels (evidence-checked).
* **per chunk** (batched per section) — a context sentence, 2–4 likely student
  questions and keywords (embedded together with the chunk to improve
  retrieval — never shown to the chatbot's LLM), a quality flag
  (`boilerplate` / `low_value` chunks are dropped) and data-quality `issues`
  for you to check (listed in the report).
* **outdated check** — chunks not tied to a program version (FAQ, intro
  text, news) are compared with the reviewed version facts from
  `programs.yaml`; contradictions are flagged `outdated` (see report →
  "Issues flagged by DeepSeek").  Version-labeled chunks are never flagged —
  the registry dates those.

DeepSeek never rewrites the stored text: the chatbot's LLM only ever sees the
page's own text plus a deterministic header.

## Metadata per chunk

| Field | Example / meaning |
|---|---|
| `source`, `filename`, `type`, `page`, `url` | used by the chatbot's citations — `filename` is the tab path, `url` opens the tab |
| `page_url`, `breadcrumb`, `section_id`, `hidden_by_default` | exact location |
| `program`, `level`, `module`, `version_label`, `accreditation_year`, `status`, `valid_for`, `version_source` | `Informatika`, `osnovne`, `OAS Informatika — akreditacija 2026`, `enrolling`, `upis od 2026/27`, `registry` |
| `doc_kind`, `generated`, `block_type` | `chunk` or `version_map` (generated overview); `text` / `table` / `table_overview` |
| `page_type`, `published_date`, `context`, `questions`, `keywords`, `quality`, `issues` | DeepSeek enrichment |
| `outdated`, `outdated_reason` | DeepSeek: contradicts the current program facts → hidden by the chatbot |
| `dedup_key`, `content_hash`, `also_on`, `prompt_version`, `llm_model`, `ingested_at` | bookkeeping |

## Tests

```bash
conda activate imi
python -m pytest
```

Offline (saved pages in `tests/fixtures`, fake DeepSeek / embedder, real Chroma in a temp dir).

The one exception is `tests/test_deepseek_live.py`, which calls the **real**
API.  It is skipped unless you opt in, so `pytest` never needs network or a key:

```bash
RUN_LIVE_LLM=1 python -m pytest tests/test_deepseek_live.py -v -s
```

Use it to check a new key or a changed `DEEPSEEK_MODEL`: it asserts the key
authenticates, the model accepts the client's combined request shape, the
reply validates against the real `EnrichmentBatch` schema, and that the
SQLite cache serves the second identical call without another API hit.
