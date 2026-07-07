"""Analytics: derive stats from saved session history."""

import json
import os
from collections import Counter
from pathlib import Path

_HISTORY_DIR = Path(os.getenv("DATA_DIR", "./data/open-webui")) / "history"


def _sessions() -> list[dict]:
    if not _HISTORY_DIR.exists():
        return []
    out = []
    for p in _HISTORY_DIR.glob("*.json"):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def get_stats() -> dict:
    sessions = _sessions()
    if not sessions:
        return {}

    total_queries = 0
    thumbs_up = thumbs_down = 0
    kb_ctr: Counter = Counter()
    model_ctr: Counter = Counter()
    best_scores: list[float] = []

    for s in sessions:
        for msg in s.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            total_queries += 1
            fb = msg.get("feedback")
            if fb == "up":
                thumbs_up += 1
            elif fb == "down":
                thumbs_down += 1
            meta = msg.get("meta", {})
            if meta.get("kb"):
                kb_ctr[meta["kb"]] += 1
            if meta.get("model"):
                model_ctr[meta["model"]] += 1
            if meta.get("scores"):
                best_scores.append(min(meta["scores"]))

    rated = thumbs_up + thumbs_down
    return {
        "total_sessions": len(sessions),
        "total_queries": total_queries,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "approval_pct": round(thumbs_up / rated * 100) if rated else None,
        "avg_best_score": round(sum(best_scores) / len(best_scores), 2)
        if best_scores
        else None,
        "kb_usage": dict(kb_ctr.most_common()),
        "model_usage": dict(model_ctr.most_common()),
    }


def recent_queries(limit: int = 50) -> list[dict]:
    """Return the most recent assistant turns across all sessions, newest first."""
    sessions = _sessions()
    rows = []
    for s in sessions:
        last_user = ""
        for msg in s.get("messages", []):
            if msg.get("role") == "user":
                last_user = msg["content"]
            elif msg.get("role") == "assistant":
                meta = msg.get("meta", {})
                rows.append(
                    {
                        "Time": (meta.get("ts") or s.get("updated_at", ""))[
                            :19
                        ].replace("T", " "),
                        "Query": last_user[:80] + ("…" if len(last_user) > 80 else ""),
                        "KB": meta.get("kb") or s.get("active_kb") or "—",
                        "Chunks": meta.get("n_chunks", "—"),
                        "Best score": round(min(meta["scores"]), 2)
                        if meta.get("scores")
                        else "—",
                        "Feedback": {"up": "👍", "down": "👎"}.get(
                            msg.get("feedback") or "", "—"
                        ),
                    }
                )
    rows.sort(key=lambda r: r["Time"], reverse=True)
    return rows[:limit]
