"""Package entry point for ``python3 -m ai_job``."""

from __future__ import annotations

from .chat_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
