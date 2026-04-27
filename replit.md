# RAG-ML Preprocessing Engine — project notes

CLI-only Python project that reads an EDA report and recommends a sklearn
preprocessing step per column, plus a free-form Q&A mode against a small
Markdown knowledge base. **No Streamlit, no Gemini, no PDFs.**

## Architecture

- **Entrypoint**: `cli.py` (argparse + interactive menu)
- **Routing**: `rag/router.py` decides between pipeline and query mode
- **Pipeline**: `rag/parser.py` → `rag/pipeline.py` → per-column
  `rag/retriever.py` + `rag/reasoning.py` + `rag/critic.py` → final report
- **Query**: `rag/query.py` does single-shot retrieval + LLM
- **Knowledge base**: `rag/knowledge.py` builds a FAISS index from
  `data/knowledge/sklearn_basics.md`, persisted to `storage/`
- **Embeddings**: `BAAI/bge-small-en-v1.5` via `sentence-transformers`
- **LLM**: Ollama HTTP (default `gemma3:1b`) with graceful fallback when
  unreachable (controlled by `LLM_GRACEFUL_FALLBACK=1`)

## Critic (5 layers)

In `rag/critic.review_decision`, applied per column:

0. Safety scrub of LLM output
1. Reject sklearn estimator names (e.g. `RandomForestRegressor`)
2. Type enforcement (numeric ↔ scaler/imputer, categorical ↔ encoder)
3. Doc re-ranking — promote preprocessing docs, demote model docs
4. Second-pass LLM validation
5. Patch row dict + audit trail in `reason`

Layers 0–3 are deterministic and run even when the LLM is offline.

## Configuration

Env vars (see `core/config.py`):

| Var | Default |
|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | `gemma3:1b` |
| `LLM_GRACEFUL_FALLBACK` | `1` |
| `MD_SOURCE` | `data/knowledge/sklearn_basics.md` |
| `STORAGE_DIR` | `storage` |

## Swap-in support

Users can drop in their own `data/knowledge/<file>.md`,
`storage/faiss.index`, and `storage/docs.pkl`.
`rag.knowledge.is_knowledge_synced()` returns true when both artefacts
exist, so the build is skipped — no source file required.

## Workflows

- **RAG Demo** — `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 python cli.py pipeline data/reports/sample_report.txt`
- **Tests** — `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 python -m pytest tests/ -q`

`LD_PRELOAD` is required on Nix/Debian so numpy/faiss find a recent
libstdc++; without it, `numpy` import fails with a `GLIBCXX_3.4.32` error.

## Tests

76 tests across 9 files in `tests/`. The suite uses a `StubRetriever`
fixture and `LLM_GRACEFUL_FALLBACK=1` so it runs offline in ~1.5s.

## Demo rendering

`docs/render_demos.py` invokes the real CLI and rasterises the output to
`docs/img/*.png` (Pillow-based terminal renderer). Re-run after any
user-visible change.

## context / total_context

LLM-friendly code-index folders. `total_context/` holds the generator
scripts (`structure_extractor.py`, `docstring_extractor.py`,
`project_summary.py`); `context/` holds their JSON/TXT outputs.

## Recent changes (2026-04-27)

- Removed all Streamlit, PDF, and Gemini code paths.
- Added env-driven `core/config.py`.
- Rewrote `cli.py` with argparse + interactive menu.
- Rewrote `rag/pipeline.py` for lazy retriever init.
- Rewrote `rag/critic.py`; fixed `_parse_critic_output` so a leading
  `Decision:` / `Final Decision:` prefix doesn't leak into the result.
- Added `tests/` (76 tests) and `docs/render_demos.py`.
- Added bundled `data/knowledge/sklearn_basics.md` and
  `data/reports/sample_report.txt` so the demo runs out of the box.
- Configured `RAG Demo` and `Tests` workflows with the libstdc++ shim.
