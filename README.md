# AI RAG LLM Demo

A fully containerized, locally-run Retrieval-Augmented Generation (RAG) Q&A application. Upload documents, ask questions, get answers grounded in your own content — with real engineering discipline behind it: monitoring, evaluation, containerization, and honest handling of failure modes, not just a notebook demo wired to an LLM.

Everything runs locally on Apple Silicon. The only native (non-Docker) service is Ollama, which needs direct Metal GPU access that Docker on macOS can't pass through.

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌──────────────────────────┐
│  Streamlit  │─────▶│   FastAPI   │─────▶│  LangGraph orchestration │
│     UI      │◀─────│   (api)     │◀─────│                          │
└─────────────┘ SSE  └──────┬──────┘      │   ┌──────────┐           │
                            │             │   │ retrieve │◀────┐     │
                            │             │   └────┬─────┘     │     │
                            │             │        ▼           │     │
                            │             │   ┌──────────┐     │     │
                            │             │   │  assess  │     │     │
                            │             │   └────┬─────┘     │     │
                            │             │        │           │     │
                            │             │   retry│           │pass │
                            │             │        ▼           │  │  │
                            │             │  ┌─────────────┐   │  │  │
                            │             │  │ reformulate │───┘  │  │
                            │             │  └─────────────┘      │  │
                            │             └───────────────────────┼──┘
                            │                                     ▼
                            ▼                                generate
                      ┌─────────────┐                              │
                      │   Qdrant    │◀───────────────────  Ollama (native,
                      │ (vectors)   │                        Metal GPU)
                      └──────▲──────┘
                             │
                      ┌──────┴──────┐      ┌─────────────┐
                      │   Dagster   │      │  Prometheus │
                      │ (ingestion) │      │  + Grafana  │◀── latency
                      └─────────────┘      │ (query-side │    metrics
                                           │  latency)   │
                                           └─────────────┘
```

**Query path:** the UI streams a question to FastAPI over SSE. A LangGraph state machine retrieves chunks from Qdrant, judges (via LLM-as-judge) whether the retrieved context is sufficient, reformulates and retries once if not, and — once judged sufficient — streams the final answer token-by-token straight from Ollama back through FastAPI to the UI.

**Ingestion path:** uploaded files are handed to a Dagster job that parses (PDF/Markdown/text), chunks, embeds (`all-MiniLM-L6-v2`), and writes vectors into Qdrant, with a consistent metadata schema (`file_name`, `content_type`, `page`, `ingested_at`) across all three file types. FastAPI polls Dagster for run completion before confirming success to the UI. Ingested files can be listed and individually deleted from the UI's sidebar.

**Monitoring:** query-side latency (retrieve/assess/reformulate/generate) is fully instrumented via Prometheus + Grafana. Ingestion-side latency currently relies on Dagster's own Run page rather than Grafana — see [Known Limitations & Future Improvements](#known-limitations--future-improvements).

## Stack

| Layer | Tech |
|---|---|
| UI | Streamlit |
| API | FastAPI |
| Query orchestration | LangGraph |
| Ingestion orchestration | Dagster |
| Ingestion/chunking | LlamaIndex, `pdfplumber` |
| Vector DB | Qdrant |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| LLM | `llama3.1:8b` via Ollama (native, Metal GPU) |
| Monitoring | Prometheus + Grafana |
| Eval | Hand-rolled LLM-as-judge |
| Infra | Docker Compose, OrbStack, `uv` |

## Setup

**Prerequisites:**
- macOS on Apple Silicon
- [Ollama](https://ollama.com) installed natively, with the model pulled: `ollama pull llama3.1:8b`
- [OrbStack](https://orbstack.dev) or Docker Desktop
- Ollama running (`ollama serve`, or it starts automatically on first `ollama run`)

**Run everything else in containers:**

```bash
docker compose up --build
```

This starts Qdrant, Dagster, the FastAPI backend, the Streamlit UI, Prometheus, and Grafana. First build will take a few minutes (pulling base images, installing dependencies).

**Access points:**
- UI: [http://localhost:8501](http://localhost:8501)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Dagster UI: [http://localhost:3000](http://localhost:3000)
- Grafana: [http://localhost:3001](http://localhost:3001)
- Prometheus: [http://localhost:9090](http://localhost:9090)

**Optional environment variables** (set in `docker-compose.yml`'s `ui` service, or your shell before running natively):
- `DEBUG_MODE` (default `false`) — controls the default state of the sidebar debug toggle, which reveals per-query path/retry/assess-reasoning info and full retrieval history.

**Maintenance:** `ingestion/ingest.py` doubles as a manual ingestion CLI and a wipe/reingest utility:

```bash
python ingest.py path/to/file.pdf                  # ingest a single file
python ingest.py -c documents -p ./files/corpus     # wipe collection, reingest a directory, restart api
python ingest.py --all                              # wipe every collection (asks to confirm)
```

`-c/--collection` and `--all` both prompt for confirmation before deleting anything. Restarting the `api` container happens automatically after a wipe, since it caches Qdrant's collection schema at startup.

## Usage

1. Open the UI and upload a document (PDF, Markdown, or plain text) via the sidebar.
2. Wait for the ingestion status to confirm success — this runs a Dagster job under the hood (parse → chunk → embed → index).
3. Ask a question in the chat box. The input disables while a response streams, to prevent submitting a second question mid-answer.
4. Each answer shows its top-3 supporting sources (by retrieval similarity, not the model's self-reported citations — see [Key Design Decisions](#key-design-decisions)).
5. Toggle **Debug mode** in the sidebar to see which path the query took (direct answer vs. retry vs. insufficient context), the assess step's reasoning, and — if a retry happened — the full retrieval history across attempts.
6. Under **Manage files** in the sidebar, see every currently-ingested file and delete any of them individually. Files with the same name ingested more than once are shown separately, disambiguated by ingestion timestamp.

## Project Structure

```
.
├── docker-compose.yml       # defines all services: qdrant, api, ui, dagster, prometheus, grafana
├── Dockerfile.api           # FastAPI service image
├── Dockerfile.ui            # Streamlit service image
├── Dockerfile.dagster       # Dagster service image
├── prometheus.yml           # Prometheus scrape config (targets the api service)
├── grafana/
│   └── provisioning/
│       ├── datasources/prometheus.yml    # auto-registers Prometheus as a datasource
│       └── dashboards/                   # dashboard.yml (provider config) + rag-latency.json
├── pyproject.toml           # single shared dependency set for all three Python services
├── uv.lock
├── config.py                # shared, env-overridable settings (Qdrant/Ollama/Dagster URLs, top-k, etc.)
│
├── api/
│   ├── main.py              # FastAPI app: /query (SSE), /ingest, /documents (list/delete), /health, /metrics
│   └── graph.py             # LangGraph state machine: retrieve → assess → reformulate loop
│
├── ingestion/
│   ├── __init__.py
│   ├── pipeline.py           # Dagster assets: raw_documents → chunks → embeddings → qdrant_index
│   ├── ingest.py             # manual ingestion CLI + wipe/reingest/restart maintenance flags
│   ├── pdf_loader.py         # PDF parsing (pdfplumber)
│   ├── md_loader.py          # Markdown parsing
│   ├── txt_loader.py         # plain text parsing
│   └── naming.py             # shared UUID-prefixed ingest naming, used by api/main.py and ingest.py
│
├── ui/
│   └── app.py                # Streamlit chat UI: streaming, debug mode, sources, upload
│
└── eval/
    ├── testset.json          # 30-question test set (see Evaluation below)
    ├── evaluate.py           # LLM-as-judge evaluation runner
    └── report/               # eval_<timestamp>.json reports land here
```

**Gitignored, bind-mounted into containers at runtime** (not part of the repo, but referenced by the services above):
- `.dagster_home/` — Dagster's local run/event storage
- `files/` — durable ingestion working directory, shared between `api` and `dagster`
- `logs/` — API logs

Grafana's Prometheus datasource and dashboard are provisioned automatically from `grafana/provisioning/` on container start — no manual UI setup required on a fresh clone.

## Evaluation

`eval/evaluate.py` runs a fixed 30-question test set (`eval/testset.json`) against the live system and scores each answer on faithfulness and correctness using a second LLM call as judge. Run it with:

```bash
python eval/evaluate.py
```

Reports are written to `eval/report/eval_<timestamp>.json`, with a summary of average scores printed to the console.

The test set was written against a specific set of sample documents (a mix of `.txt`, `.md`, and `.pdf` files) used during development. Those sample files are gitignored and not included in this repo, so the test set won't produce meaningful results without first ingesting equivalent documents yourself — it's included as a demonstration of the evaluation approach, not a ready-to-run benchmark for a fresh clone.

## Key Design Decisions

**LLM-as-judge over cheap retrieval signals.** Four cheap signals — cross-encoder reranking, similarity score-gap, chunk redundancy, keyword overlap — were tested at n=20 and none reliably predicted whether retrieved context was sufficient. An LLM-as-judge step (`assess`) replaced them, at the cost of one extra Ollama call per query.

**Top-3-by-similarity as displayed citations, not the model's self-report.** Early versions asked the generation model to self-report which sources it used via a `SOURCES:` tag; spot-checking showed roughly 25% of self-reported citations pointed at chunks that didn't actually support the claim. Since Qdrant already ranks retrieved chunks by similarity, the top-3 by that score are shown instead — a signal the retrieval system itself produces, not one the generation model has to get right.

**Generation runs outside the LangGraph state machine.** LangGraph nodes return a complete state update, which is incompatible with token-by-token streaming to the UI. The graph itself only covers `retrieve → assess → reformulate` looping and the terminal `insufficient_context` path; once `assess` signals a pass, FastAPI calls Ollama's streaming API directly rather than routing generation through another graph node.

**Durable ingestion temp files, not `tempfile.TemporaryDirectory`.** An earlier design tied uploaded files' lifetime to FastAPI's own polling loop via a temp directory. Since Dagster's daemon runs as a separate process, it could attempt to read a file after FastAPI had already cleaned it up. Uploads are now written to a durable `files/ingest/` directory with a UUID-prefixed filename (to avoid collisions between concurrent uploads sharing a name) and only deleted once Dagster confirms the run reached a terminal state.

**Prometheus needs a named volume.** Without one, Prometheus's own metric history was wiped on every container recreation — including the ones triggered by a Mac going to sleep. A named volume for the TSDB fixed this.

**Two-phase question submission to block double-submits.** Streamlit reruns the entire script on any new interaction, tearing down whatever was previously running — including a script that's mid-way through streaming an answer. Naively setting `chat_input(disabled=...)` doesn't help, because that state is only reflected the *next* time the script runs; the widget stays visually enabled for the entire duration of the in-flight answer. The fix splits submission into two script runs: the first stashes the question and immediately reruns to flush a disabled input to the browser, and only the second run makes the actual (slow) backend call. This shrinks the window during which the input still looks submittable from the full length of a response down to roughly one rerun's flush time.

**One consistent metadata schema across all file types, not per-loader shapes.** PDF, Markdown, and plain-text ingestion originally produced three different payload shapes in Qdrant — plain text in particular went through LlamaIndex's default `SimpleDirectoryReader` metadata (`file_path`, `file_size`, etc.) with none of the project's own fields. This worked fine until a feature (listing and deleting ingested files) needed to group and sort chunks into files consistently regardless of type. Rather than special-case three shapes inside that endpoint, every loader now writes the same four keys (`file_name`, `content_type`, `page`, `ingested_at`), with `page: None` where the concept doesn't apply (Markdown, plain text).

**UUID-prefixed ingest identity, shared via one helper, not duplicated.** Every ingest — whether via API upload or the manual CLI script — gets a `{uuid4().hex}_{filename}` identity, so two files with the same name never collide in Qdrant and can be deleted independently. The two-line generation logic lives in one place (`ingestion/naming.py`) rather than being copied into both call sites, because a downstream regex (the UI's display-name stripping) depends on the exact prefix format; two independently-drifting copies would be a silent, hard-to-notice failure mode.

## Known Limitations & Future Improvements

- **Monitoring is split across two tools.** Query-side latency (retrieve/assess/reformulate/generate) is fully covered in Grafana. Ingestion-side latency (parsing/chunking/embedding/indexing) has no Prometheus instrumentation; Dagster's own Run page covers per-run debugging instead. This is a deliberate scope decision: unifying it would mean either scraping Dagster's process directly (unreliable for short-lived jobs — the metrics can be gone before a scrape interval catches them) or adding a Pushgateway container, and neither was justified without a concrete need to trend ingestion latency over time.
- **A small race window remains around double-submitting a question.** The two-phase submit (above) shrinks the window during which a second question could be submitted while the first is still streaming down to roughly one Streamlit rerun's flush time, but doesn't eliminate it — under enough network latency it's still possible to slip a second submission in. If that happens, the in-flight answer is lost and the app shows a warning rather than corrupting or silently dropping state.
- **No true query cancellation or ingestion abort.** There's currently no way to stop an in-flight query or a running ingestion job partway through — you can block a *new* question from starting while one is in progress (see above), but not stop the one that's already running. For queries specifically, this is a UX cost (waiting out a 30-50s response), not a correctness risk — the double-submit fix above already prevents the failure mode where an abandoned query's orphaned backend call could interfere with the next one. Real cancellation would require converting the query graph's nodes to async so an in-flight Ollama call actually gets interrupted, not just the UI's connection to it; ingestion abort would call Dagster's run-termination API. Both are backlogged, not scheduled — revisit if either becomes a repeatedly felt need.
- **Borderless/multi-page PDF tables aren't extracted correctly.** `pdfplumber`'s line-based table strategy only handles ruled/bordered tables; unruled tables fall back to garbled inline text rather than structured extraction. Would need a layout-aware extraction model or custom multi-page stitching; not planned unless eval data shows this materially hurting answer quality on table-heavy sources.
- **Mandarin Chinese isn't supported.** A nice-to-have for a future iteration, not currently scheduled.
