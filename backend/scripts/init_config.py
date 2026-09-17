#!/usr/bin/env python3
"""Create local configuration without overwriting existing secrets (stdlib only)."""

import secrets
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        raise SystemExit(".env already exists; keeping the existing configuration")
    text = (root / ".env.example").read_text(encoding="utf-8")
    text = text.replace("CHANGE_ME_API_KEY", secrets.token_urlsafe(32))
    text = text.replace("CHANGE_ME_DB_PASSWORD", secrets.token_urlsafe(32))
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    target.chmod(0o600)
    provider = "unknown"
    for line in text.splitlines():
        if line.startswith("VISION_PROVIDER="):
            provider = line.split("=", 1)[1].strip()
    print(f"Created .env with random API and database credentials. VISION_PROVIDER={provider}.")
    if provider == "local":
        print(
            "Recognition runs locally on open models. Docker Compose fetches the "
            "weights on first start; outside Docker run: python scripts/fetch_models.py"
        )
    elif provider == "mock":
        print("Mock mode returns no detections; set VISION_PROVIDER=local for recognition.")


if __name__ == "__main__":
    main()
