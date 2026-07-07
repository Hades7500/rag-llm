import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

_HISTORY_DIR = Path(os.getenv("DATA_DIR", "./data/open-webui")) / "history"


def _dir() -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return _HISTORY_DIR


# ── Public API ────────────────────────────────────────────────────────────────


def new_id() -> str:
    return uuid.uuid4().hex


def save(session_id: str, messages: list[dict], active_kb: str | None) -> None:
    """Persist a session to disk. Safe to call on every assistant turn."""
    if not messages:
        return

    path = _dir() / f"{session_id}.json"

    created_at = datetime.now(timezone.utc).isoformat()
    if path.exists():
        try:
            created_at = json.loads(path.read_text(encoding="utf-8")).get(
                "created_at", created_at
            )
        except Exception:
            pass

    data = {
        "id": session_id,
        "title": _title(messages),
        "created_at": created_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "active_kb": active_kb,
        "messages": messages,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load(session_id: str) -> dict | None:
    """Return session dict or None if not found / corrupt."""
    path = _dir() / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_all(limit: int = 40) -> list[dict]:
    """Return session summaries sorted by most-recently-updated first."""
    summaries = []
    for p in _dir().glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", ""),
                    "active_kb": data.get("active_kb"),
                    "count": len(data.get("messages", [])),
                }
            )
        except Exception:
            pass
    summaries.sort(key=lambda x: x["updated_at"], reverse=True)
    return summaries[:limit]


def delete(session_id: str) -> None:
    path = _dir() / f"{session_id}.json"
    if path.exists():
        path.unlink()


# ── Internal helpers ──────────────────────────────────────────────────────────


def _title(messages: list[dict]) -> str:
    """Use first user message (truncated) as the session title."""
    for m in messages:
        if m.get("role") == "user":
            text = m["content"].strip().replace("\n", " ")
            return text[:60] + ("…" if len(text) > 60 else "")
    return "New chat"


def fmt_date(iso: str) -> str:
    """Human-readable relative date label."""
    try:
        dt = datetime.fromisoformat(iso).astimezone()
        now = datetime.now(dt.tzinfo)
        delta = now - dt
        if delta.days == 0:
            return dt.strftime("Today %H:%M")
        if delta.days == 1:
            return dt.strftime("Yesterday %H:%M")
        if delta.days < 7:
            return dt.strftime("%A %H:%M")
        return dt.strftime("%b %d")
    except Exception:
        return ""
