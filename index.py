"""Vercel entrypoint: the FastAPI app, plus the built dashboard it serves.

Vercel's FastAPI preset routes every request to this application and gives it the
Python function budget (500 MB uncompressed) that the legacy `/api` directory does not.
The React build therefore has to be served from here rather than by the static CDN: the
`assets` directory is mounted as-is and any other path returns `index.html`, which is
what the client-side router needs for a deep link such as `/students/MAT-0001`.

`app.main` is unchanged and still owns startup; this module only decides *how* the
process is reached. Locally the same build is served by Vite's dev server, so the static
branches below are inactive outside Vercel.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app, bootstrap  # noqa: E402  (path set immediately above)

DIST = ROOT / "frontend" / "dist"

# Import time, not first request: a cold instance that cannot find its artifacts should
# fail the way the local server does - before it answers anything.
app.state.models = bootstrap()


if DIST.is_dir():
    # Vite writes content-hashed files under `assets`, so they are safe to serve with a
    # long cache; the entry document is not, and is returned with no-store.
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def dashboard(path: str) -> FileResponse:
        """Serve a built file when it exists, else the single-page app's shell.

        Declared after the routers, so `/api`, `/docs` and `/openapi.json` keep their
        own handlers; everything else is either a real file in the build or a route
        the client-side router owns (such as `/students/MAT-0001`).
        """
        candidate = (DIST / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(
            DIST / "index.html",
            headers={"Cache-Control": "no-store"},
        )
