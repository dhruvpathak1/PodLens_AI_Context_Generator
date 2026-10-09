<p align="center"><img src="public/favicon.svg" width="104" alt="PodLens logo"></p>

# PodLens: AI-Powered Context for Your Podcasts 🎙️✨

PodLens is an AI-powered context engine for podcasts. It transcribes an episode, identifies the people, companies, places and technologies mentioned, and surfaces each one as a source card with a summary, map and image, timed to the moment it is spoken. It can also build a chronological timeline of the episode and answer questions about it, with citations that jump to the exact moment.

**▶ Try the live demo:** https://podlens.dhruvpathak.com (no sign-up, no API keys; see [Live Demo](#-live-demo))

## 🧰 Built With

| Category | Technologies |
|---|---|
| **Languages** | [![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org) [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org) [![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/docs/Web/HTML) [![CSS](https://img.shields.io/badge/CSS-663399?style=for-the-badge&logo=css&logoColor=white)](https://developer.mozilla.org/docs/Web/CSS) [![YAML](https://img.shields.io/badge/YAML-CB171E?style=for-the-badge&logo=yaml&logoColor=white)](https://yaml.org) [![Bash](https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnubash&logoColor=white)](https://www.gnu.org/software/bash/) |
| **Frontend** | [![React 19](https://img.shields.io/badge/React%2019-61DAFB?style=for-the-badge&logo=react&logoColor=000000)](https://react.dev) [![Vite](https://img.shields.io/badge/Vite-9135FF?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev) [![Web Audio API](https://img.shields.io/badge/Web%20Audio%20API-30363d?style=for-the-badge)](https://developer.mozilla.org/docs/Web/API/Web_Audio_API) [![Inter (Google Fonts)](https://img.shields.io/badge/Inter%20%28Google%20Fonts%29-4285F4?style=for-the-badge&logo=googlefonts&logoColor=white)](https://fonts.google.com/specimen/Inter) |
| **Backend** | [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com) [![Uvicorn](https://img.shields.io/badge/Uvicorn-30363d?style=for-the-badge)](https://www.uvicorn.org) [![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev) [![HTTPX](https://img.shields.io/badge/HTTPX-30363d?style=for-the-badge)](https://www.python-httpx.org) [![python-dotenv](https://img.shields.io/badge/python--dotenv-ECD53F?style=for-the-badge&logo=dotenv&logoColor=000000)](https://github.com/theskumar/python-dotenv) |
| **Speech & audio** | [![OpenAI Whisper](https://img.shields.io/badge/OpenAI%20Whisper-412991?style=for-the-badge)](https://github.com/openai/whisper) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org) [![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org) |
| **NLP & entities** | [![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io) [![Claude (Anthropic)](https://img.shields.io/badge/Claude%20%28Anthropic%29-D97757?style=for-the-badge&logo=claude&logoColor=white)](https://www.anthropic.com/claude) |
| **LLM & AI** | [![LangChain](https://img.shields.io/badge/LangChain-7FC8FF?style=for-the-badge&logo=langchain&logoColor=000000)](https://www.langchain.com) [![OpenAI GPT](https://img.shields.io/badge/OpenAI%20GPT-412991?style=for-the-badge)](https://platform.openai.com/docs/models) [![OpenAI Embeddings](https://img.shields.io/badge/OpenAI%20Embeddings-412991?style=for-the-badge)](https://platform.openai.com/docs/guides/embeddings) |
| **Data sources** | [![Wikipedia](https://img.shields.io/badge/Wikipedia-000000?style=for-the-badge&logo=wikipedia&logoColor=white)](https://www.wikipedia.org) [![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-7EBC6F?style=for-the-badge&logo=openstreetmap&logoColor=white)](https://www.openstreetmap.org) [![Unsplash](https://img.shields.io/badge/Unsplash-000000?style=for-the-badge&logo=unsplash&logoColor=white)](https://unsplash.com/developers) |
| **Testing & quality** | [![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://docs.pytest.org) [![ESLint](https://img.shields.io/badge/ESLint-4B32C3?style=for-the-badge&logo=eslint&logoColor=white)](https://eslint.org) [![TypeScript strict](https://img.shields.io/badge/TypeScript%20strict-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/tsconfig#strict) |
| **DevOps & tooling** | [![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com) [![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions) [![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-222222?style=for-the-badge&logo=githubpages&logoColor=white)](https://pages.github.com) [![Node.js](https://img.shields.io/badge/Node.js-5FA04E?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org) [![npm](https://img.shields.io/badge/npm-CB3837?style=for-the-badge&logo=npm&logoColor=white)](https://www.npmjs.com) [![Git](https://img.shields.io/badge/Git-F03C2E?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com) |
| **Live mode (parked)** | [![OpenAI Realtime](https://img.shields.io/badge/OpenAI%20Realtime-412991?style=for-the-badge)](https://platform.openai.com/docs/guides/realtime) [![WebSockets](https://img.shields.io/badge/WebSockets-30363d?style=for-the-badge)](https://developer.mozilla.org/docs/Web/API/WebSockets_API) [![MediaRecorder](https://img.shields.io/badge/MediaRecorder-30363d?style=for-the-badge)](https://developer.mozilla.org/docs/Web/API/MediaRecorder) |

Also used: asyncio, python-multipart, `typescript-eslint`, `eslint-plugin-react-hooks`, Docker Compose, `concurrently`, and Python venv. Data is stored as JSON entity exports and timestamped `.txt` transcripts.

### See it in Action
https://github.com/user-attachments/assets/4b8ee1bf-2946-4abc-87b3-8432aa8f52d6

https://github.com/user-attachments/assets/d97e2ba6-fbba-426e-b249-aaa80cc6fb22

---

## 📑 Contents

- [Built With](#-built-with)
- [Key Features](#-key-features)
- [How It Works](#%EF%B8%8F-how-it-works)
- [AI Entity Review](#-ai-entity-review)
- [Episode Timeline](#-episode-timeline)
- [Ask the Episode (LLM Q&A)](#-ask-the-episode-llm-qa)
- [Choosing Your Models: spaCy vs Claude vs OpenAI](#-choosing-your-models-spacy-vs-claude-vs-openai)
- [APIs and Services Used](#-apis-and-services-used)
- [PodLens REST API](#-podlens-rest-api)
- [Project Structure](#%EF%B8%8F-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the App](#-running-the-app)
- [Live Demo](#-live-demo)
- [Configuration Reference](#%EF%B8%8F-configuration-reference)
- [Roadmap](#%EF%B8%8F-roadmap)

---

## 🚀 Key Features

| Feature | What it does | Powered by |
|---|---|---|
| **Transcription** | Timestamped transcript of any uploaded episode, run locally | OpenAI Whisper (local) |
| **Entity tagging** | Finds people, companies, places, technologies and events | spaCy (local) or Anthropic Claude |
| **Disambiguation** | Tells "Apple" the company from the fruit, "Amazon" the company from the river | Custom context rules |
| **AI entity review** | Fixes misheard names, merges duplicates, drops vague tags, and removes summaries or photos that don't match the episode | LangChain + OpenAI |
| **Source cards** | Summary, map and photo for every entity, shown as playback reaches it | Wikipedia, OpenStreetMap, Unsplash |
| **Episode timeline** | Every dated event mentioned, in date order, each with a link to the moment it is said | LangChain + OpenAI |
| **Ask the episode** | Grounded Q&A with citations that seek the player to the quote | LangChain + OpenAI (+ embeddings) |
| **Synced transcript** | Waveform scrubber; the active line highlights during playback; click any line to seek; search | React, Web Audio |
| **Dashboard** | Six panels on a dotted dark canvas, each scrolling on its own; intro cards on first load; light theme available | CSS design tokens |
| **Free public demo** | Pre-processed sample episodes on GitHub Pages, with no server and no API keys | Static JSON + audio |

---

## ⚙️ How It Works

```
 Upload audio ─▶ FFmpeg ─▶ Whisper ─▶ Transcript + segments
                                          │
                                          ▼
                              NER (spaCy or Claude) ─▶ Entities with timestamps
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
   Source cards                   Episode timeline              Ask the episode
   Wikipedia · OSM · Unsplash     LangChain + OpenAI            LangChain + OpenAI
              │                           │                           │
              └──────────────▶ React dashboard synced to playback ◀───┘
```

---

## 🔎 AI Entity Review

Speech recognition and entity taggers make predictable mistakes on podcasts. A review step checks every entity against the episode before (and after) the source cards are built.

| Problem | Example | What the review does |
|---|---|---|
| Misheard name | "Traderus 8", "Apollo Alto Garage" | **Fix**: "Traitorous Eight", "Palo Alto" (and the right type) |
| Same thing, several names | "SF", "San Francisco", "San Francisco, California" | **Merge** into one entity |
| Not a real entity | "San", the show's own name, filler | **Drop** |
| Wrong sense on Wikipedia | "Yerba Buena" resolves to the mint plant | Writes a precise search query ("Yerba Buena San Francisco history"), then **removes** any summary or photo that still doesn't match |

**How it works**

1. **Before lookups**: each unique entity is sent with the transcript lines it appears in and the opening of the episode. The model returns one verdict per entity (`keep`, `fix`, `merge`, `drop`) plus a disambiguated Wikipedia search query, using strict structured output.
2. **Lookups**: Wikipedia, maps and photos use the corrected names and search queries.
3. **After lookups**: the model compares each card's Wikipedia summary and photo description with how the episode uses the name, and strict mismatches are removed from the card.

The Entities panel shows the result ("checked by AI, 6 corrected, 5 removed"; hover for the full list). The review **fails open**: if the model is unavailable, cards are built from the original entities. Turn it off with `ENTITY_REVIEW=false`; demo builds accept `--no-review`.

---

## 🧭 Episode Timeline

One click turns the episode into a chronological timeline of every company, person, product, deal and event mentioned, with real-world dates where the speakers give them.

**How it works**

- The transcript is split into windows (`TIMELINE_WINDOW_CHARS`, default 24,000 characters) so long episodes fit the model's context.
- Windows are processed concurrently (`TIMELINE_MAX_CONCURRENCY`, default 4). Each window only receives the entities tagged inside it.
- LangChain's `with_structured_output` uses OpenAI **strict JSON schema** mode, so every event is guaranteed to match the schema; no regex parsing of model output.
- Results from all windows are merged, de-duplicated and clamped to valid timestamps. One failed window does not fail the whole timeline.
- In the UI, **Create timeline** lists every event in date order on a quiet date spine: the date on the left (in the accent colour), the event title and a **Played at** link to the moment it is mentioned on the right. Undated events come last; the model's description shows when you hover a title.

**Event schema**

```ts
type TimelineEvent = {
  id: string
  title: string
  description: string
  category: 'COMPANY' | 'PERSON' | 'PRODUCT' | 'TECHNOLOGY' | 'EVENT' | 'PLACE' | 'DEAL' | 'OTHER'
  date_label: string | null   // as stated: "1993", "late 1990s"
  year: number | null
  month: number | null
  mentioned_at_sec: number    // where it is first mentioned in the episode
  entities: string[]
  importance: 1 | 2 | 3       // passing mention, notable, central
}
```

**Request**

```bash
curl -X POST http://127.0.0.1:8000/api/timeline \
  -H "Content-Type: application/json" \
  -d '{
        "segments": [{"id": 0, "start": 12.0, "end": 23.0,
                      "text": "In 1966 researchers began studying lions in the Serengeti."}],
        "entities": [{"type": "PLACE", "text": "Serengeti", "start_sec": 13, "end_sec": 15, "chunk_id": 0}],
        "source_label": "Lions 101.mp3"
      }'
```

**Response (abridged)**

```json
{
  "model": "gpt-5.4-mini",
  "windows": 1,
  "failed_windows": 0,
  "events": [
    {
      "id": "ev0",
      "title": "Long-term Serengeti lion study begins",
      "category": "EVENT",
      "date_label": "1966",
      "year": 1966,
      "mentioned_at_sec": 12.0,
      "entities": ["Serengeti"],
      "importance": 3
    }
  ]
}
```

---

## 💬 Ask the Episode (LLM Q&A)

Ask anything about the episode and get an answer **grounded only in the transcript**, with citations you can click to hear the exact quote.

**Hybrid context strategy**

| Transcript length | Mode | What is sent to the model |
|---|---|---|
| Up to `ASK_FULL_CONTEXT_CHARS` (default 60,000 characters) | `full` | The whole timestamped transcript |
| Longer | `retrieval` | Top-K (`ASK_TOP_K`, default 8) overlapping ~60 s chunks, ranked by OpenAI embeddings |

- Embeddings for a long episode are computed **once** and cached in memory, keyed by a fingerprint of the transcript, so follow-up questions are fast and cheap.
- The last 6 turns of the conversation are sent with each question, so follow-ups like "when did he say that?" work.
- **Citation validation**: every quote the model returns is checked against the transcript. Citations that do not match are dropped, and an answer whose citations all fail is flagged `unverified: true`.
- If the episode does not cover the question, the model says so (`found: false`) instead of guessing.

**Request**

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Who wrote The Serengeti Lion?",
        "segments": [{"id": 2, "start": 24.0, "end": 35.0,
                      "text": "George Schaller published The Serengeti Lion in 1972."}],
        "history": []
      }'
```

**Response**

```json
{
  "answer": "George Schaller, who published The Serengeti Lion in 1972.",
  "found": true,
  "citations": [{ "start_sec": 24.0, "quote": "George Schaller published The Serengeti Lion in 1972." }],
  "mode": "full",
  "model": "gpt-5.4-mini",
  "unverified": false
}
```

---

## 🧪 Choosing Your Models: spaCy vs Claude vs OpenAI

PodLens uses three model families, each for a different job:

| Job | Options | Configured by |
|---|---|---|
| Speech to text | Whisper `tiny` to `large` (local) | `WHISPER_MODEL` |
| Entity tagging (NER) | spaCy `sm` / `md` / `lg` (local) or Claude (API) | `ENTITY_BACKEND`, `SPACY_MODEL`, `CLAUDE_MODEL` |
| Entity review | OpenAI chat model (API) | `OPENAI_REVIEW_MODEL` |
| Timeline and Q&A | OpenAI chat + embedding models (API) | `OPENAI_TIMELINE_MODEL`, `OPENAI_ASK_MODEL`, `OPENAI_EMBED_MODEL` |

> OpenAI does not tag entities. It reviews what spaCy or Claude found (fix, merge, drop) and powers the timeline and Q&A.

### Entity tagging: spaCy vs Claude

| | spaCy `en_core_web_sm` | spaCy `md` / `lg` | Claude (`claude-haiku-4-5`) |
|---|---|---|---|
| Runs | Locally | Locally | Anthropic API |
| Cost | Free | Free | Paid per token |
| Speed | Fastest | Fast | Slower (network, batched) |
| Privacy | Audio and text stay on your machine | Same | Transcript text is sent to Anthropic |
| Messy podcast speech | Misses or confuses entities | Noticeably better | Best: handles slang, partial names, context |
| Setup | Installed with requirements | `python -m spacy download en_core_web_md` | `ANTHROPIC_API_KEY` |

**How the backend is chosen**: `ENTITY_BACKEND` wins if set. If it is unset and `ANTHROPIC_API_KEY` is present, Claude is used; otherwise spaCy. The UI can force one with `VITE_ENTITY_BACKEND`.

### Whisper model sizes

| Model | Parameters | Use when |
|---|---|---|
| `tiny` | 39M | Quick tests, low-RAM machines |
| `base` (default) | 74M | Good balance on a laptop CPU |
| `small` | 244M | Noticeably cleaner text, which also improves NER |
| `medium` / `large` | 769M / 1.55B | Best accuracy; slow on CPU, best with a GPU |

---

## 🌐 APIs and Services Used

| Service | Used for | Endpoint | Key needed |
|---|---|---|---|
| **OpenAI Whisper** (local) | Speech to text | Runs in-process | No |
| **spaCy** (local) | Entity tagging | Runs in-process | No |
| **Anthropic Messages API** | Entity tagging (Claude backend) | `api.anthropic.com` via the `anthropic` SDK | `ANTHROPIC_API_KEY` |
| **OpenAI Chat Completions** | Entity review, timeline and Q&A (strict JSON schema) | via `langchain-openai` | `OPENAI_API_KEY` |
| **OpenAI Embeddings** | Retrieval for long episodes in Ask | via `langchain-openai` | `OPENAI_API_KEY` |
| **Wikipedia Action API** | Find the best article for an entity | `en.wikipedia.org/w/api.php` | No |
| **Wikipedia REST API** | Summary and thumbnail | `en.wikipedia.org/api/rest_v1/page/summary` | No |
| **Nominatim (OpenStreetMap)** | Geocoding places | `nominatim.openstreetmap.org/search` | No (set `NOMINATIM_USER_AGENT`) |
| **OpenStreetMap embed** | Interactive map on place cards | `openstreetmap.org/export/embed.html` | No |
| **Unsplash Search API** | Photo for each card (with photographer credit) | `api.unsplash.com/search/photos` | `UNSPLASH_ACCESS_KEY` |

Every external service is optional except Whisper and spaCy. Missing keys disable only the feature that needs them, and `GET /api/health` reports which features are active.

---

## 🔌 PodLens REST API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status and which optional features are configured |
| `POST` | `/api/transcribe` | Upload audio (multipart); returns transcript, segments and tagged entities |
| `POST` | `/api/extract-entities` | Tag entities in existing transcript segments |
| `POST` | `/api/enrich-entities` | Review entities with AI (when `segments` are sent) and build source cards |
| `POST` | `/api/timeline` | Chronological timeline of everything mentioned |
| `POST` | `/api/ask` | Grounded Q&A with citations |

Interactive docs are available at `http://127.0.0.1:8000/docs` while the API is running.

**Transcribe an episode**

```bash
curl -X POST http://127.0.0.1:8000/api/transcribe \
  -F "audio=@episode.mp3" \
  -F "extract_entities=true" \
  -F "entity_backend=spacy"     # optional: spacy | claude
```

```json
{
  "transcript": "Lions are the only truly social big cats...",
  "segments": [{ "id": 0, "start": 0.0, "end": 11.0, "text": "Lions are the only truly social big cats..." }],
  "saved_path": ".../server/transcripts/episode_20261008T221500Z.txt",
  "document": {
    "backend": "spacy",
    "entities": [{ "type": "PLACE", "text": "Serengeti", "start_sec": 13.0, "end_sec": 15.0, "chunk_id": 1 }]
  },
  "entity_error": null
}
```

**Build source cards**

```bash
curl -X POST http://127.0.0.1:8000/api/enrich-entities \
  -H "Content-Type: application/json" \
  -d '{"entities": [{"type": "PLACE", "text": "Serengeti", "start_sec": 13, "end_sec": 15, "chunk_id": 1}]}'
```

**Check which features are on**

```bash
curl http://127.0.0.1:8000/api/health
# {"ok":true,"model":"base","entity_backend":"spacy","unsplash_configured":true,
#  "openai_configured":true,"entity_review":true,"live_mode":false,"realtime_transcription":false}
```

---

## 🗂️ Project Structure

```
├── src/                         React frontend
│   ├── app/                     App.tsx (composition only: wires hooks, lays out the 3 columns)
│   ├── api/                     One module per backend endpoint (fetch + response parsing)
│   ├── config/env.ts            Build-time settings (API base URL, entity backend)
│   ├── types/                   Shared data shapes (transcript, entities, timeline, ask)
│   ├── shared/                  Panel shell, formatting helpers, icons, Alert, Skeleton
│   ├── features/                One folder per feature: hooks, components and helpers
│   │   ├── episode/             Upload -> transcribe -> tag -> enrich pipeline state
│   │   ├── playback/            Audio player, waveform scrubber, playback clock
│   │   ├── transcript/          Transcript panel: synced text, search, sentence timing
│   │   ├── entities/            Live card grid, Entities (NER) panel, type filter
│   │   ├── about/               PodLens panel: project explanation, samples or upload
│   │   ├── timeline/            Episode timeline
│   │   ├── ask/                 Ask the episode chat
│   │   ├── demo/                Demo mode: sample picker, pre-computed timeline/Ask
│   │   ├── theme/ upload/       Theme toggle, drop zone
│   │   └── live/                PARKED live microphone mode (not imported; see its README)
│   └── styles/                  Global CSS split by area; index.css imports them in order
├── public/demo/                 Pre-processed demo episodes served by the public site
│
└── server/                      FastAPI backend
    ├── main.py                  Entry point shim (`uvicorn main:app`)
    ├── app/
    │   ├── main.py              create_app(): CORS, routers, optional static UI
    │   ├── core/                Settings (every env var in one place) and logging
    │   ├── api/                 HTTP layer: routes/, request schemas, error helpers
    │   ├── services/            Business logic: audio, transcription, episode, storage,
    │   │                        entities/ (spaCy or Claude NER), enrichment/ (Wikipedia,
    │   │                        Nominatim, Unsplash), entity_review.py, source_cards.py,
    │   │                        timeline.py, ask.py, demo_bundle.py
    │   └── live/                PARKED live endpoints, mounted only with ENABLE_LIVE_MODE=true
    ├── scripts/build_demo.py    CLI: turn an audio file into a demo episode (npm run demo:build)
    └── tests/                   pytest suite
```

**Design rules**: routes stay thin (validate, call a service, map errors to HTTP codes); services know nothing about HTTP; heavy libraries (Whisper, LangChain) load lazily so the API starts instantly; on the frontend, `App.tsx` only composes feature hooks and components.

---

## 📦 Installation and Setup

### Prerequisites
- **Node.js** 18+
- **Python** 3.9+
- **FFmpeg** on your `PATH` (`brew install ffmpeg` on macOS)

### 1. Clone and install the frontend
```bash
git clone https://github.com/dhruvpathak1/PodLens_AI_Context_Generator.git
cd PodLens_AI_Context_Generator
npm install
```

### 2. Set up the backend
The npm scripts expect the virtual environment at the project root (`venv/`).
```bash
python -m venv venv
venv/bin/python -m pip install -r server/requirements.txt   # Windows: venv\Scripts\python
```

### 3. Configure
```bash
cp .env.example .env
```
Then fill in what you need:

| Variable | Needed for |
|---|---|
| `UNSPLASH_ACCESS_KEY` | Photos on source cards |
| `ANTHROPIC_API_KEY` | Claude entity tagging |
| `OPENAI_API_KEY` | AI entity review, Episode timeline and Ask the episode |
| `NOMINATIM_USER_AGENT` | Your app name or contact email (requested by OpenStreetMap) |

---

## 🚀 Running the App

### Development
```bash
npm run dev
```
- **Frontend:** http://localhost:5173
- **Backend API:** http://127.0.0.1:8000 (docs at `/docs`)

The first transcription downloads the Whisper model, which can take a few minutes.

### Checks
```bash
npm run check        # typecheck + lint + backend tests
npm run build        # production bundle in dist/
```
Backend tests need the dev requirements once: `venv/bin/python -m pip install -r server/requirements-dev.txt`.

### 🐳 Docker
```bash
docker compose up --build
```
Open http://localhost:8080. One container serves the built UI and the API; data persists in the `podlens-data` volume. See [DOCKER.md](DOCKER.md) for sharing and deployment.

---

## 🎧 Live Demo

The public site runs on GitHub Pages in **demo mode**: it plays episodes that were processed once, ahead of time, so it needs **no server, no API keys, and costs nothing** to run. Visitors get the full experience (synced transcript, live source cards, timeline, and answers with clickable citations) instantly.

**How it works**

```
 Your machine (once per episode)                    GitHub Pages (every visitor)
 ─────────────────────────────────                  ────────────────────────────
 npm run demo:build ─▶ Whisper ─▶ NER ─▶ AI review  public/demo/index.json
                   ─▶ cards ─▶ timeline ─▶ Q&A ───▶  public/demo/<slug>/episode.json
                   ─▶ compressed audio              public/demo/<slug>/audio.mp3
```

- The demo build sets `VITE_DEMO_MODE=true`. The PodLens panel shows sample episodes instead of the upload box, and the Timeline and Ask panels answer from the stored results. The rest of the UI is exactly the same.
- Each sample ships with pre-computed answers to a few questions. Typed questions are matched to them loosely; anything else gets a short reply listing the questions that do have answers, as clickable links.
- No request ever goes to an API, so there is nothing to abuse and no bill.

**Add a demo episode**

```bash
npm run demo:build -- path/to/episode.mp3 \
  --title "Apollo 11: The Landing" \
  --description "Mission audio from the first crewed Moon landing." \
  --credit "Audio: NASA (public domain)"

git add public/demo && git commit -m "demo: add Apollo 11 episode" && git push
```

| Flag | Purpose |
|---|---|
| `--title` | Name shown in the picker (required) |
| `--description` | One line under the title |
| `--credit` | Audio attribution shown under the sample buttons |
| `--question "..."` | Preset Ask question (repeatable); default: generated from the top entities |
| `--backend spacy\|claude` | Entity tagger for this build |
| `--skip-timeline`, `--skip-ask` | Build without OpenAI (free, local tools only) |
| `--no-review` | Skip the AI entity review (on by default with an OpenAI key) |
| `--bitrate 64k` | MP3 bitrate of the published audio (about 0.5 MB per minute) |

The build uses your local `.env` keys **once**, on your machine; the keys are never written to the output. Without `OPENAI_API_KEY` the episode is still built, just without the timeline and preset answers.

> **Only publish audio you own or that is licensed for redistribution** (your own recordings, public-domain sources such as NASA or LibriVox, or Creative Commons with attribution). The audio is served publicly from the site.

---

## ⚙️ Configuration Reference

All settings live in `.env` (see [`.env.example`](.env.example) for the full, commented list).

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Whisper size: `tiny`, `base`, `small`, `medium`, `large` |
| `ENTITY_BACKEND` | auto | `spacy` or `claude` |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy pipeline for the local backend |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model for NER |
| `OPENAI_TIMELINE_MODEL` | `gpt-5.4-mini` | Model for the timeline |
| `TIMELINE_WINDOW_CHARS` | `24000` | Transcript window size per timeline call |
| `TIMELINE_MAX_CONCURRENCY` | `4` | Parallel timeline windows |
| `OPENAI_ASK_MODEL` | `gpt-5.4-mini` | Model for Ask |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embeddings for long-episode retrieval |
| `ASK_FULL_CONTEXT_CHARS` | `60000` | Above this, Ask switches to retrieval |
| `ASK_TOP_K` | `8` | Chunks retrieved per question |
| `ENTITY_REVIEW` | `true` | AI review of entities and source cards (needs `OPENAI_API_KEY`) |
| `OPENAI_REVIEW_MODEL` | `gpt-5.4-mini` | Model for the entity review |
| `CORS_EXTRA_ORIGINS` | none | Extra allowed browser origins (comma-separated) |
| `VITE_TRANSCRIBE_URL` | dev proxy | API URL for production frontend builds |
| `ENABLE_LIVE_MODE` | `false` | Mounts the parked live endpoints |
| `VITE_DEMO_MODE` | `false` (`true` on GitHub Pages) | Play pre-processed episodes from `public/demo/`; no API calls |

---

## 🗺️ Roadmap

- **Live mode**: real-time microphone transcription via OpenAI Realtime. The code is parked, documented and compiling in `src/features/live/` and `server/app/live/`; see the README in each folder to re-enable it.
- **Non-blocking uploads**: move transcription off the event loop so the API stays responsive during long jobs.
- **Persistent vector store**: replace the in-memory embedding cache (for example, pgvector) so Ask survives restarts.
- **Episode library**: save processed episodes and reopen them without re-transcribing.
- **Authentication** before any public deployment.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ for the podcasting community.*
