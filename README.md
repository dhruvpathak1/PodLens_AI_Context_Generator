# PodLens: AI-Powered Context for Your Podcasts 🎙️✨

PodLens is a full-stack, real-time context generator that transforms audio into an interactive, enriched experience. By combining high-accuracy transcription with intelligent Named Entity Recognition (NER) and parallel web enrichment, PodLens provides listeners with instant deep-dives into the people, places, and technologies mentioned in any discussion.

![PodLens Architecture](public/architecture_diagram.png)

## 🚀 Key Features

- **Real-Time Live Mode:** Capture microphone input in 10s chunks with an ordered processing pipeline. Transcript and context cards appear dynamically as you speak.
- **Whisper-Powered Transcription:** Local execution of OpenAI's Whisper model for precise, timestamped transcripts without cloud costs.
- **Intelligent NER & Disambiguation:** Custom logic to distinguish between homonyms (e.g., "Apple" the company vs. "apple" the fruit) using conversational context.
- **Multi-Source Enrichment:**
  - **Wikipedia:** Instant summaries and thumbnails for entities.
  - **OpenStreetMap:** Interactive map embeds for every mentioned location.
  - **Unsplash:** Beautiful, high-quality photography to visualize abstract concepts.
- **Modern Dashboard:** A sleek, responsive React interface with "Source Cards" that roll on and off the screen in sync with the audio.

### See it in Action
https://github.com/user-attachments/assets/4b8ee1bf-2946-4abc-87b3-8432aa8f52d6
  
https://github.com/user-attachments/assets/d97e2ba6-fbba-426e-b249-aaa80cc6fb22

---

## 🛠️ Technology Stack

### Frontend
- ![React](https://img.shields.io/badge/React_19-20232A?style=flat&logo=react&logoColor=61DAFB) **React 19** with Concurrent Mode for smooth UI updates.
- ![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white) Typed for reliability and developer velocity.
- ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white) Lightning-fast HMR and optimized builds.
- ![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white) Utility-first styling for a custom, modern look.

### Backend
- ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi) High-performance Python framework.
- ![Python](https://img.shields.io/badge/Python_3.9+-3776AB?style=flat&logo=python&logoColor=white) Powering the heavy-lifting NLP tasks.
- ![OpenAI Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?style=flat&logo=openai&logoColor=white) Local SOTA audio-to-text.
- ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=flat&logo=spacy&logoColor=white) Industrial-strength NLP for local entity extraction.

---

## 🏗️ Technical Architecture

PodLens uses a sophisticated asynchronous pipeline to ensure low-latency context delivery:

1. **Audio Ingest**: The React frontend captures audio via `MediaRecorder` (Live) or `File API`.
2. **FastAPI Processing**: Audio is sent to the backend where it's normalized via `FFmpeg` and passed to **Whisper**.
3. **NER Engine**:
   - **Local Path**: Uses `spaCy` for fast, private extraction.
   - **Cloud Path (Optional)**: Leverages **Anthropic's Claude** for high-accuracy extraction using custom system prompts.
   - **Context Disambiguation**: A custom layer analyzes surrounding keywords to correctly tag entities (e.g., identifying "Amazon" as a company when "AWS" is mentioned).
4. **Parallel Enrichment**: Using Python's `asyncio`, PodLens fires concurrent requests to Wikipedia, Nominatim, and Unsplash. Results are cached and streamed back to the frontend.
5. **Dynamic UI**: React merges incoming chunks into a unified session, managing "Source Cards" that appear and disappear based on the current playback timestamp.

---

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
```bash
cd server
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cd ..
```

### 3. Configuration
Copy `.env.example` to `.env` and configure your API keys:
- `UNSPLASH_ACCESS_KEY`: Required for photo enrichment.
- `ANTHROPIC_API_KEY`: Optional for Claude-powered NER.

---

## 🚀 Running the App

### Development Mode
You can run both the frontend and the backend simultaneously from the root directory:

```bash
npm run dev
```

- **Frontend:** [http://localhost:5173](http://localhost:5173)
- **Backend API:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

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
