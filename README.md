# Noise

Cut through the noise. Get answers.

Noise is a question-answering app that takes a natural-language question, searches the web, scrapes top results, and uses an LLM pipeline to return a single concise answer backed by cited sources — no tab-hopping required.

## How It Works

```
User question
     │
     ▼
  FastAPI backend
     │
     ├─ analyze_question     → understand intent & plan search
     ├─ reason_and_act       → web search (Serper) + page scraping (loop if needed)
     ├─ filter_evidence      → drop irrelevant content
     ├─ aggregate_answer     → claim extraction → evidence ranking → consensus → final write
     └─ format_response      → structured JSON back to the client
```

The backend uses **LangGraph** to orchestrate a stateful graph of nodes. Each node is a step in the pipeline; conditional edges control looping (fetch more evidence) and short-circuiting (no relevant docs → skip synthesis).

The aggregation layer:
1. **Claim Extractor** — pulls discrete factual claims from scraped sources
2. **Evidence Ranker** — scores each claim against the original question
3. **Consensus Builder** — groups agreeing claims, flags contradictions & uncertainty
4. **Final Writer** — produces the answer with inline source citations

## Tech Stack

| Layer | Tech |
|-------|------|
| Mobile | React Native (Expo SDK 54), iOS/Android |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| LLM orchestration | LangChain, LangGraph |
| LLM providers | OpenAI, Google Gemini, Groq (configurable) |
| Web search | Serper API |
| Embeddings / retrieval | FAISS, LangChain embeddings |
| Config | Pydantic Settings + `.env` |

## Project Structure

```
├── App.js                  # React Native entry point
├── backend/
│   └── app/
│       ├── main.py         # FastAPI app factory
│       ├── config.py       # env-based settings (LLM provider, API keys)
│       ├── api/routes.py   # /api/ask, /api/ask/full endpoints
│       ├── graph/          # LangGraph workflow, nodes, state
│       ├── chains/         # LLM wrappers, RAG pipeline
│       ├── aggregation/    # claim extraction, ranking, consensus, final writer
│       ├── retrieval/      # embeddings, FAISS store, document ingestion
│       ├── tools/          # web search (Serper), page scraper
│       └── models/         # Pydantic schemas
└── ios/                    # Xcode project (managed by Expo)
```

## Getting Started

### Prerequisites

- Node.js & npm
- Python 3.11+
- An API key for at least one LLM provider (OpenAI / Gemini / Groq)
- A [Serper](https://serper.dev) API key for web search

### Backend

```bash
cd backend
pip install -e ".[dev]"

# create .env in backend/
cat <<EOF > .env
LLM_PROVIDER=groq          # or openai, gemini
GROQ_API_KEY=your-key
SERPER_API_KEY=your-key
EOF

uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. Hit `/health` to verify.

### Mobile App

```bash
npm install
npx expo start
```

Scan the QR code or press `i` for iOS simulator / `a` for Android emulator.

### Running Tests

```bash
cd backend
pytest
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/ask` | POST | Returns `{ answer, model, sources, snippets }` |
| `/api/ask/full` | POST | Returns the full aggregated result (claims, consensus, evidence, answer) |

Request body for both `/ask` endpoints:

```json
{ "question": "your question here" }
```

## What's Not Included (Yet)

- Auth / user accounts
- Conversation history or memory
- Streaming responses
- Eval / benchmarking framework
