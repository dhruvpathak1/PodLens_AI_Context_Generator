# Run PodLens with Docker

One container serves the **built React UI** and the **FastAPI** backend (Whisper, NER, enrichment).

## Quick start

```bash
docker compose up --build
```

Open **http://localhost:8080**. The first transcription downloads the Whisper model — expect a long delay and **several GB** of disk/RAM (CPU image).

## Optional configuration

- Copy `.env.example` to `.env` next to `docker-compose.yml` and fill in keys. Docker Compose reads that file for `${VAR}` substitution and forwards common variables into the container.
- Override CORS for another browser origin:  
  `CORS_EXTRA_ORIGINS=https://example.com docker compose up`

## Share with others

1. **Same LAN**: expose port `8080` (or change the mapping in `docker-compose.yml`) and share your machine IP.
2. **Internet**: run the same compose stack on a small VPS, open firewall port **8080** (or put **nginx + HTTPS** in front).
3. **Publish an image** (GitHub Container Registry example):

   ```bash
   docker build -t ghcr.io/YOUR_USER/podlens:latest .
   docker push ghcr.io/YOUR_USER/podlens:latest
   ```

   Others run:

   ```bash
   docker run --rm -p 8080:8000 -v podlens-data:/data ghcr.io/YOUR_USER/podlens:latest
   ```

## GPU (optional)

The default image uses **CPU** Whisper. For NVIDIA GPUs you’d extend the Dockerfile with CUDA base images and compatible PyTorch wheels; not included here to keep the shareable image simple.

## Troubleshooting

- **`docker compose` fails during `pip install`**: ensure the build has network access; `server/requirements.txt` installs spaCy and downloads models.
- **Container exits / OOM**: Whisper + PyTorch need RAM; try `WHISPER_MODEL=tiny` in `.env` for a smaller model.
