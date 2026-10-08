# PodLens: AI-Powered Context for Your Podcasts 🎙️✨

PodLens is a full-stack, real-time context generator that transforms audio into an interactive, enriched experience. By combining high-accuracy transcription with intelligent Named Entity Recognition (NER) and parallel web enrichment, PodLens provides listeners with instant deep-dives into the people, places, and technologies mentioned in any discussion.

![PodLens Architecture](public/architecture_diagram.png)

## 🚀 Key Features

- **Episode Timeline:** One click builds a chronological timeline of everything mentioned (LangChain + OpenAI structured output). Events reveal in the right panel as playback reaches them.
- **Ask the Episode:** Grounded Q&A over the transcript with clickable citations that jump to the exact moment.
- **Whisper-Powered Transcription:** Local execution of OpenAI's Whisper model for precise, timestamped transcripts without cloud costs.
- **Intelligent NER & Disambiguation:** Custom logic to distinguish between homonyms (e.g., "Apple" the company vs. "apple" the fruit) using conversational context.
- **Multi-Source Enrichment:**
  - **Wikipedia:** Instant summaries and thumbnails for entities.
  - **OpenStreetMap:** Interactive map embeds for every mentioned location.
  - **Unsplash:** Beautiful, high-quality photography to visualize abstract concepts.
- **Modern Dashboard:** A responsive React interface with a synced transcript, light/dark theme, and "Source Cards" that roll on and off the screen in sync with the audio.
- **Live Mode (future scope):** Real-time microphone transcription is parked in its own module. See [Live Mode](#-live-mode-future-scope).

### See it in Action
https://github.com/user-attachments/assets/4b8ee1bf-2946-4abc-87b3-8432aa8f52d6
  
https://github.com/user-attachments/assets/d97e2ba6-fbba-426e-b249-aaa80cc6fb22

---

## 🛠️ Technology Stack

### Frontend
- ![React](https://img.shields.io/badge/React_19-20232A?style=flat&logo=react&logoColor=61DAFB) **React 19** with Concurrent Mode for smooth UI updates.
- ![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white) Typed for reliability and developer velocity.
- ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white) Lightning-fast HMR and optimized builds.
- ![CSS](https://img.shields.io/badge/CSS-1572B6?style=flat&logo=css3&logoColor=white) Plain CSS with design tokens and a light/dark theme.

### Backend
- ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi) High-performance Python framework.
- ![Python](https://img.shields.io/badge/Python_3.9+-3776AB?style=flat&logo=python&logoColor=white) Powering the heavy-lifting NLP tasks.
- ![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?style=flat&logo=openai&logoColor=white) Local SOTA audio-to-text.
- ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white) Industrial-strength NLP for local entity extraction.
- ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white) Structured LLM output for the timeline and Ask (OpenAI).

---

## 🏗️ Technical Architecture

PodLens uses a sophisticated asynchronous pipeline to ensure low-latency context delivery:

1. **Audio Ingest**: The user uploads an audio file (MP3, WAV, M4A, etc.) from the React frontend.
2. **FastAPI Processing**: Audio is sent to the backend where it's normalized via `FFmpeg` and passed to **Whisper**.
3. **NER Engine**:
   - **Local Path**: Uses `spaCy` for fast, private extraction.
   - **Cloud Path (Optional)**: Leverages **Anthropic's Claude** for high-accuracy extraction using custom system prompts.
   - **Context Disambiguation**: A custom layer analyzes surrounding keywords to correctly tag entities (e.g., identifying "Amazon" as a company when "AWS" is mentioned).
4. **Parallel Enrichment**: Using Python's `asyncio`, PodLens fires concurrent requests to Wikipedia, Nominatim, and Unsplash. Source cards are returned to the frontend as one batch.
5. **Dynamic UI**: "Source Cards" appear and disappear based on the current playback timestamp.
6. **AI Layer (optional)**: With `OPENAI_API_KEY` set, `/api/timeline` and `/api/ask` add the episode timeline and grounded Q&A.

## 🗂️ Project Structure

```
├── src/                         React frontend
│   ├── app/                     App.tsx (composition only: wires hooks, lays out the 3 columns)
│   ├── api/                     One module per backend endpoint (fetch + response parsing)
│   ├── config/env.ts            Build-time settings (API base URL, entity backend)
│   ├── types/                   Shared data shapes (transcript, entities, timeline, ask)
│   ├── shared/                  Formatting helpers, icons, Alert, Skeleton
│   ├── features/                One folder per feature: hooks, components and helpers
│   │   ├── episode/             Upload -> transcribe -> tag -> enrich pipeline state
│   │   ├── playback/            Audio player and playback clock
│   │   ├── transcript/          Left sidebar: synced transcript, search, sentence timing
│   │   ├── entities/            Live cards, entity filter, source cards
│   │   ├── rail/                Right "Episode AI" panel (Timeline / Ask tabs)
│   │   ├── timeline/            Episode timeline
│   │   ├── ask/                 Ask the episode chat
│   │   ├── theme/ upload/       Theme toggle, drop zone
│   │   └── live/                PARKED live microphone mode (not imported; see its README)
│   └── styles/                  Global CSS split by area; index.css imports them in order
│
└── server/                      FastAPI backend
    ├── main.py                  Entry point shim (`uvicorn main:app`)
    ├── app/
    │   ├── main.py              create_app(): CORS, routers, optional static UI
    │   ├── core/                Settings (all env vars in one place) and logging
    │   ├── api/                 HTTP layer: routes/, request schemas, error helpers
    │   ├── services/            Business logic: audio, transcription, episode, storage,
    │   │                        entities/ (spaCy or Claude NER), enrichment/ (Wikipedia,
    │   │                        Nominatim, Unsplash), timeline.py, ask.py
    │   └── live/                PARKED live endpoints, mounted only with ENABLE_LIVE_MODE=true
    └── tests/                   pytest suite (no network or API keys needed)
```

Rules of thumb: routes stay thin (validate, call a service, map errors to HTTP codes); services
know nothing about HTTP; on the frontend, `App.tsx` only composes feature hooks and components.

## 🔌 API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Status and which optional features are configured |
| POST | `/api/transcribe` | Upload audio; returns transcript, segments and tagged entities |
| POST | `/api/extract-entities` | Tag entities in existing transcript segments |
| POST | `/api/enrich-entities` | Build source cards (Wikipedia, map, photo) |
| POST | `/api/timeline` | Chronological timeline of everything mentioned (OpenAI) |
| POST | `/api/ask` | Grounded Q&A over the episode, with citations (OpenAI) |

## 🎙️ Live Mode (Future Scope)

Live microphone transcription is parked. The code is kept, documented and compiling in
`src/features/live/` and `server/app/live/`, but is not wired into the app. See the README in
each folder for how to re-enable it.

## 📦 Installation & Setup

### Prerequisites
- **Node.js** (v18+)
- **Python** (3.9+)
- **FFmpeg** (System-level installation required for Whisper)

### 1. Clone & Install
```bash
git clone https://github.com/dhruvpathak1/PodLens_AI_Context_Generator.git
cd PodLens_AI_Context_Generator
npm install
```

### 2. Backend Setup
The npm scripts expect the virtual environment at the project root (`venv/`).
```bash
python -m venv venv
venv/bin/python -m pip install -r server/requirements.txt   # Windows: venv\Scripts\python
```

### 3. Configuration
Copy `.env.example` to `.env` and configure your API keys:
- `UNSPLASH_ACCESS_KEY`: Required for photo enrichment.
- `ANTHROPIC_API_KEY`: Optional for Claude-powered NER.
- `OPENAI_API_KEY`: Optional; enables the Episode Timeline and Ask the Episode.
- `NOMINATIM_USER_AGENT`: Your app name or contact email (requested by OpenStreetMap).

---

## 🚀 Running the App

### Development Mode
You can run both the frontend and the backend simultaneously from the root directory:

```bash
npm run dev
```

- **Frontend:** [http://localhost:5173](http://localhost:5173)
- **Backend API:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

### ✅ Checks
```bash
npm run check        # typecheck + lint + backend tests
npm run build        # production bundle in dist/
```
Backend tests need the dev requirements once: `venv/bin/python -m pip install -r server/requirements-dev.txt`.

### 🐳 Docker Support
PodLens is fully containerized. You can run the entire stack with a single command:

```bash
docker compose up --build
```
*Note: The first run will download the Whisper model (~150MB - 3GB depending on config). See [DOCKER.md](DOCKER.md) for more details.*

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ for the podcasting community.*
