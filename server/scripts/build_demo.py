"""Turn an audio file into a pre-processed demo episode for the public (static) site.

Usage (from the project root, with the venv set up as in the README):

    npm run demo:build -- path/to/episode.mp3 --title "Apollo 11: The Landing" \
        --description "Mission audio from the first Moon landing." \
        --credit "Audio: NASA (public domain)"

This runs the real pipeline once on your machine (Whisper, entity tagging, source cards, and,
if OPENAI_API_KEY is set, the timeline and preset Q&A) and writes:

    public/demo/index.json
    public/demo/<slug>/episode.json
    public/demo/<slug>/audio.mp3

Commit `public/demo/` and push: the GitHub Pages build serves it with no API and no keys.
Only publish audio you own or that is licensed for redistribution.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `python scripts/build_demo.py` as well as `python -m scripts.build_demo`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import configure_logging  # noqa: E402
from app.services.demo_bundle import DEFAULT_OUT_DIR, DemoOptions, build_demo_episode, slugify  # noqa: E402


def parse_args(argv: list[str] | None = None) -> DemoOptions:
    """Parse command-line flags into `DemoOptions`."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="Audio file (mp3, wav, m4a, webm, mp4, ...)")
    p.add_argument("--title", required=True, help="Episode title shown in the demo picker")
    p.add_argument("--slug", help="Folder name under public/demo (default: from the title)")
    p.add_argument("--description", default="", help="One-line description for the picker")
    p.add_argument("--credit", default="", help='Attribution, e.g. "Audio: NASA (public domain)"')
    p.add_argument(
        "--question", action="append", default=[], dest="questions",
        help="Preset Ask question (repeatable). Default: generated from the top entities.",
    )  # fmt: skip
    p.add_argument("--backend", choices=["spacy", "claude"], help="Entity backend (default: server setting)")
    p.add_argument("--language", help="ISO language code to skip Whisper's detection, e.g. en")
    p.add_argument("--bitrate", default="64k", help="MP3 bitrate for the published audio (default 64k)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output folder (default public/demo)")
    p.add_argument("--skip-timeline", action="store_true", help="Do not call OpenAI for the timeline")
    p.add_argument("--skip-ask", action="store_true", help="Do not call OpenAI for preset answers")
    p.add_argument(
        "--no-review", action="store_true",
        help="Skip the LLM review of entities and source cards (on by default with an OpenAI key)",
    )  # fmt: skip
    a = p.parse_args(argv)

    if not a.audio.is_file():
        p.error(f"audio file not found: {a.audio}")
    return DemoOptions(
        audio_path=a.audio.resolve(),
        title=a.title.strip(),
        slug=slugify(a.slug or a.title),
        description=a.description.strip(),
        credit=a.credit.strip(),
        questions=[q.strip() for q in a.questions if q.strip()],
        backend=a.backend,
        language=a.language,
        bitrate=a.bitrate,
        out_dir=a.out.resolve(),
        skip_timeline=a.skip_timeline,
        skip_ask=a.skip_ask,
        review=False if a.no_review else None,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    configure_logging()
    opts = parse_args(argv)
    try:
        asyncio.run(build_demo_episode(opts))
    except Exception as e:  # show a clean message instead of a traceback for expected failures
        print(f"Demo build failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
