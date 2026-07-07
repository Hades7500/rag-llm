import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Make sibling modules importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent))

import analytics
import history
from chroma_utils import (
    collection_count,
    delete_collection,
    get_client,
    get_or_create_collection,
    list_collections,
    query_collection,
    upsert_documents,
)
from doc_utils import chunk_text, make_ids, parse_file
from ollama_utils import OllamaEmbedding, get_models, is_embed_model, stream_chat

# ── Configuration from environment ───────────────────────────────────────────

OLLAMA_HOST = (
    os.getenv("OLLAMA_BASE_URLS", "http://127.0.0.1:11434")
    .split(";")[0]
    .strip()
    .rstrip("/")
)
CHROMA_HOST = os.getenv("CHROMA_HTTP_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_HTTP_PORT", "8000"))
EMBED_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

_data_dir = Path(os.getenv("DATA_DIR", "./data/open-webui"))
_data_dir.mkdir(parents=True, exist_ok=True)
SYSTEM_PROMPT_FILE = _data_dir / "system_prompt.md"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "When a CONTEXT section is provided, prefer that information in your answer. "
    "If the context does not fully cover the question, supplement it with your own knowledge. "
    "Always give a useful answer."
)


def _build_rag_system(base_prompt: str, context_block: str) -> str:
    """Inject retrieved context into the system prompt."""
    return (
        f"{base_prompt}\n\n"
        "---\n"
        "CONTEXT (from the knowledge base):\n"
        f"{context_block}\n"
        "---\n"
        "Use the CONTEXT above to answer. "
        "If the context covers the question, prefer it. "
        "If it does not fully cover the question, use your own knowledge to fill the gaps."
    )


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Tighten sidebar width */
        section[data-testid="stSidebar"] { min-width: 300px; max-width: 360px; }

        /* Status dots */
        .dot-ok  { color: #22c55e; font-size: 1rem; }
        .dot-err { color: #ef4444; font-size: 1rem; }

        /* KB meta line */
        .kb-meta { font-size: 0.8rem; color: #9ca3af; margin-top: -0.5rem; }

        /* Hide the default Streamlit footer */
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cached resources (one instance per server process) ───────────────────────


@st.cache_resource
def _chroma_client():
    return get_client(CHROMA_HOST, CHROMA_PORT)


@st.cache_resource
def _embed_fn():
    return OllamaEmbedding(model=EMBED_MODEL, base_url=OLLAMA_HOST)


# ── Session-state helpers ─────────────────────────────────────────────────────


def _load_system_prompt() -> str:
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


def _save_system_prompt(text: str) -> None:
    SYSTEM_PROMPT_FILE.write_text(text, encoding="utf-8")


def _init_state() -> None:
    defaults: dict = {
        "session_id": history.new_id(),
        "messages": [],
        "active_kb": None,
        "system_prompt": _load_system_prompt(),
        "chat_model": DEFAULT_CHAT_MODEL,
        "rag_enabled": True,
        "top_k": 3,
        "score_threshold": 1.5,
        "max_context_chars": 2000,
        "view": "chat",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _new_chat() -> None:
    """Reset to a blank conversation without touching settings."""
    st.session_state.session_id = history.new_id()
    st.session_state.messages = []


def _load_session(session_id: str) -> None:
    """Restore a saved session into state."""
    data = history.load(session_id)
    if not data:
        return
    st.session_state.session_id = data["id"]
    st.session_state.messages = data["messages"]
    if data.get("active_kb"):
        st.session_state.active_kb = data["active_kb"]


def _autosave() -> None:
    """Persist the current conversation after every assistant turn."""
    history.save(
        st.session_state.session_id,
        st.session_state.messages,
        st.session_state.active_kb,
    )


def _retrieval_query(messages: list[dict], current_prompt: str, window: int = 3) -> str:
    """Build a ChromaDB retrieval query from recent history + current message."""
    prior_user = [m["content"] for m in messages[:-1] if m["role"] == "user"][-window:]
    parts = prior_user + [current_prompt]
    return " ".join(parts)


# ── Service-health checks (cached 10 s so every keystroke doesn't ping) ──────


@st.cache_data(ttl=10)
def _ollama_ok() -> bool:
    import requests

    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2).raise_for_status()
        return True
    except Exception:
        return False


@st.cache_data(ttl=10)
def _chroma_ok() -> bool:
    try:
        c = _chroma_client()
        c.list_collections()
        return True
    except Exception:
        return False


@st.cache_data(ttl=30)
def _fetch_models() -> list[str]:
    return get_models(OLLAMA_HOST)


# ── Small helpers ─────────────────────────────────────────────────────────────


def _get_collections() -> list[str]:
    try:
        return list_collections(_chroma_client())
    except Exception:
        return []


def _open_collection(name: str):
    return get_or_create_collection(_chroma_client(), name, _embed_fn())


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 Local RAG")
        st.divider()

        # ── View toggle ──────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "💬 Chat",
                use_container_width=True,
                type="primary" if st.session_state.view == "chat" else "secondary",
            ):
                st.session_state.view = "chat"
                st.rerun()
        with c2:
            if st.button(
                "📊 Analytics",
                use_container_width=True,
                type="primary" if st.session_state.view == "analytics" else "secondary",
            ):
                st.session_state.view = "analytics"
                st.rerun()
        st.divider()

        # ── Chat History ──────────────────────────────────────────────────
        st.markdown("### 💬 Chat History")

        if st.button("✏️ New Chat", use_container_width=True, type="primary"):
            _new_chat()
            st.rerun()

        sessions = history.list_all()
        if sessions:
            st.markdown("")
            for s in sessions:
                is_current = s["id"] == st.session_state.session_id
                kb_tag = f" · `{s['active_kb']}`" if s["active_kb"] else ""
                date_tag = history.fmt_date(s["updated_at"])
                subtitle = f"{s['count'] // 2 or 1} exchange(s){kb_tag} · {date_tag}"

                col1, col2 = st.columns([5, 1])
                with col1:
                    label = ("▶ " if is_current else "") + s["title"]
                    if st.button(
                        label,
                        key=f"hist_{s['id']}",
                        use_container_width=True,
                        help=subtitle,
                        type="primary" if is_current else "secondary",
                    ):
                        if not is_current:
                            _load_session(s["id"])
                            st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{s['id']}", help="Delete"):
                        history.delete(s["id"])
                        if is_current:
                            _new_chat()
                        st.rerun()
        else:
            st.caption("No history yet. Start chatting!")

        st.divider()

        # ── Knowledge Base ────────────────────────────────────────────────
        st.markdown("### 🗂 Knowledge Base")

        collections = _get_collections()
        CREATE_OPT = "➕  Create new domain…"
        options = collections + [CREATE_OPT]

        if st.session_state.active_kb in collections:
            current_idx = collections.index(st.session_state.active_kb)
        else:
            current_idx = len(options) - 1
            st.session_state.active_kb = None

        selected = st.selectbox(
            "Active domain",
            options=options,
            index=current_idx,
            label_visibility="collapsed",
        )

        if selected == CREATE_OPT:
            with st.form("new_kb_form", clear_on_submit=True):
                new_name = st.text_input(
                    "Domain name",
                    placeholder="e.g. medical, legal, finance",
                )
                if st.form_submit_button("✅ Create", use_container_width=True):
                    safe = new_name.strip().lower().replace(" ", "_")
                    if not safe:
                        st.error("Enter a name.")
                    else:
                        try:
                            _open_collection(safe)
                            st.session_state.active_kb = safe
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Error: {exc}")
        else:
            st.session_state.active_kb = selected
            if selected:
                try:
                    col_obj = _open_collection(selected)
                    n = collection_count(col_obj)
                    st.markdown(
                        f'<p class="kb-meta">📄 {n} chunk{"s" if n != 1 else ""} stored</p>',
                        unsafe_allow_html=True,
                    )
                except Exception:
                    pass

                if st.button(
                    "🗑 Delete this domain",
                    use_container_width=True,
                    type="secondary",
                ):
                    try:
                        delete_collection(_chroma_client(), selected)
                        st.session_state.active_kb = None
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        st.divider()

        # ── Upload Files ──────────────────────────────────────────────────
        st.markdown("### 📁 Add Documents")

        active_kb = st.session_state.active_kb
        if not active_kb:
            st.info("Select or create a domain first.")
        else:
            uploaded = st.file_uploader(
                "Drop files here",
                type=["pdf", "txt", "md", "docx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded:
                if st.button(
                    f"⬆ Add to **{active_kb}**",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        col_obj = _open_collection(active_kb)
                        embed_fn = _embed_fn()
                        bar = st.progress(0, text="Starting…")
                        total_files = len(uploaded)
                        total_chunks = 0
                        EMBED_BATCH = 10

                        for file_idx, uf in enumerate(uploaded):
                            print(
                                f"[DBG] ── {uf.name} ({len(uf.getvalue()) / 1024:.1f} KB) ──",
                                file=sys.stderr,
                                flush=True,
                            )

                            t0 = time.time()
                            raw = uf.read()
                            mock = types.SimpleNamespace(
                                name=uf.name,
                                read=lambda _raw=raw: _raw,
                            )
                            text = parse_file(mock)
                            print(
                                f"[DBG] parse:   {time.time() - t0:.2f}s  ({len(text)} chars)",
                                file=sys.stderr,
                                flush=True,
                            )

                            t1 = time.time()
                            chunks = chunk_text(text)
                            print(
                                f"[DBG] chunk:   {time.time() - t1:.2f}s  ({len(chunks)} chunks)",
                                file=sys.stderr,
                                flush=True,
                            )

                            if chunks:
                                ids = make_ids(uf.name, raw, len(chunks))
                                metas = [
                                    {"source": uf.name, "chunk": j}
                                    for j in range(len(chunks))
                                ]

                                # ── Step 1: embed all chunks (few large calls) ──
                                all_embeddings: list = []
                                n_embed_batches = max(
                                    1,
                                    (len(chunks) + EMBED_BATCH - 1) // EMBED_BATCH,
                                )
                                t_embed_total = time.time()
                                for b in range(n_embed_batches):
                                    s = b * EMBED_BATCH
                                    e = min(s + EMBED_BATCH, len(chunks))
                                    t2 = time.time()
                                    all_embeddings.extend(embed_fn(chunks[s:e]))
                                    print(
                                        f"[DBG] embed batch {b + 1}/{n_embed_batches} (chunks {s + 1}-{e}): {time.time() - t2:.2f}s",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    done = file_idx + (b + 1) / n_embed_batches * 0.9
                                    bar.progress(
                                        done / total_files,
                                        text=f"{uf.name} — embedding {e}/{len(chunks)} chunks…",
                                    )
                                print(
                                    f"[DBG] embed total: {time.time() - t_embed_total:.2f}s",
                                    file=sys.stderr,
                                    flush=True,
                                )

                                # ── Step 2: single upsert, no re-embedding ──────
                                bar.progress(
                                    (file_idx + 0.95) / total_files,
                                    text=f"{uf.name} — storing in knowledge base…",
                                )
                                t3 = time.time()
                                upsert_documents(
                                    col_obj,
                                    chunks,
                                    ids,
                                    metas,
                                    embeddings=all_embeddings,
                                )
                                print(
                                    f"[DBG] upsert:  {time.time() - t3:.2f}s",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                total_chunks += len(chunks)

                        bar.progress(1.0, text="Done!")
                        time.sleep(0.6)
                        bar.empty()
                        st.success(
                            f"Added {total_files} file(s) → {total_chunks} chunks into **{active_kb}**"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")

        st.divider()

        # ── System Prompt ─────────────────────────────────────────────────
        st.markdown("### 📝 System Prompt")

        prompt_file = st.file_uploader(
            "Load from .txt / .md file",
            type=["txt", "md"],
            key="prompt_file_uploader",
        )
        if prompt_file is not None:
            st.session_state.system_prompt = prompt_file.read().decode(
                "utf-8", errors="replace"
            )

        edited_prompt = st.text_area(
            "Prompt content",
            value=st.session_state.system_prompt,
            height=180,
            label_visibility="collapsed",
            key="prompt_textarea",
        )

        if st.button("💾 Save system prompt", use_container_width=True):
            st.session_state.system_prompt = edited_prompt
            _save_system_prompt(edited_prompt)
            st.success("Saved to disk.")

        st.divider()

        # ── Settings ──────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")

        all_models = _fetch_models()
        chat_models = [m for m in all_models if not is_embed_model(m)] or all_models

        if chat_models:
            idx = 0
            if st.session_state.chat_model in chat_models:
                idx = chat_models.index(st.session_state.chat_model)
            st.session_state.chat_model = st.selectbox(
                "Chat model",
                options=chat_models,
                index=idx,
            )
        else:
            st.warning("No models found.\n\n`ollama pull llama3.2:3b`")

        st.session_state.rag_enabled = st.toggle(
            "Use knowledge base",
            value=st.session_state.rag_enabled,
            help="Retrieve relevant document chunks and add them to the prompt.",
        )

        if st.session_state.rag_enabled:
            st.session_state.top_k = st.slider(
                "Context chunks (top-K)",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Fewer chunks = less context = faster responses.",
            )
            st.session_state.score_threshold = st.slider(
                "Similarity threshold",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.score_threshold,
                step=0.05,
                help="L2 distance cutoff. Lower = only very close matches. "
                "Raise if too few chunks are retrieved.",
            )
            st.session_state.max_context_chars = st.slider(
                "Max context chars",
                min_value=200,
                max_value=6000,
                value=st.session_state.max_context_chars,
                step=100,
                help="Hard cap on total context sent to the model. "
                "Fewer chars = faster generation.",
            )


# ── Main chat area ────────────────────────────────────────────────────────────


def _render_feedback_buttons(msg_index: int) -> None:
    """Show 👍/👎 under a specific assistant message."""
    msg = st.session_state.messages[msg_index]
    fb = msg.get("feedback")
    c1, c2, _ = st.columns([1, 1, 10])
    with c1:
        if st.button(
            "👍" if fb != "up" else "✅",
            key=f"fb_up_{msg_index}",
            help="Good answer",
            type="primary" if fb == "up" else "secondary",
        ):
            msg["feedback"] = None if fb == "up" else "up"
            _autosave()
            st.rerun()
    with c2:
        if st.button(
            "👎" if fb != "down" else "❌",
            key=f"fb_dn_{msg_index}",
            help="Poor answer",
            type="primary" if fb == "down" else "secondary",
        ):
            msg["feedback"] = None if fb == "down" else "down"
            _autosave()
            st.rerun()


def _render_chat() -> None:
    kb = st.session_state.active_kb or "none"
    model = st.session_state.chat_model
    rag_label = "on" if st.session_state.rag_enabled else "off"

    st.markdown(
        f"**KB:** `{kb}` &nbsp;·&nbsp; "
        f"**Model:** `{model}` &nbsp;·&nbsp; "
        f"**RAG:** `{rag_label}`"
    )
    st.divider()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_feedback_buttons(i)

    # Chat input (pinned to bottom by Streamlit).
    if prompt := st.chat_input("Ask a question…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context_chunks: list[str] = []
            scores: list[float] = []
            kb_active = st.session_state.rag_enabled and bool(
                st.session_state.active_kb
            )

            # ── RAG retrieval ─────────────────────────────────────────────
            if kb_active:
                try:
                    col_obj = _open_collection(st.session_state.active_kb)
                    retrieval_q = _retrieval_query(st.session_state.messages, prompt)
                    context_chunks, scores = query_collection(
                        col_obj,
                        retrieval_q,
                        n_results=st.session_state.top_k,
                        score_threshold=st.session_state.score_threshold,
                    )
                except Exception as exc:
                    st.warning(f"RAG retrieval skipped: {exc}")

            # ── Build system prompt ───────────────────────────────────────
            if context_chunks:
                context_block = "\n\n---\n\n".join(context_chunks)
                cap = st.session_state.max_context_chars
                if len(context_block) > cap:
                    context_block = context_block[:cap].rsplit(" ", 1)[0] + " …"
                system = _build_rag_system(
                    st.session_state.system_prompt, context_block
                )
                st.caption(
                    f"📎 {len(context_chunks)} chunk(s) · "
                    f"scores: {[round(s, 2) for s in scores]} · "
                    f"{len(context_block)} chars"
                )
            else:
                # No context (RAG off, no KB, or no matching chunks) — answer freely.
                if kb_active:
                    st.caption(
                        "💡 No matching documents found — answering from general knowledge."
                    )
                system = st.session_state.system_prompt

            # ── Always call the model ─────────────────────────────────────
            try:
                full_response = st.write_stream(
                    stream_chat(
                        model=st.session_state.chat_model,
                        messages=st.session_state.messages,
                        system=system,
                        base_url=OLLAMA_HOST,
                    )
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "meta": {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "model": st.session_state.chat_model,
                            "kb": st.session_state.active_kb,
                            "n_chunks": len(context_chunks),
                            "scores": scores,
                        },
                    }
                )
                _autosave()
                _render_feedback_buttons(len(st.session_state.messages) - 1)
            except Exception as exc:
                err = str(exc)
                st.error(f"Chat error: {err}")
                if "does not support" in err.lower() or "chat" in err.lower():
                    st.info(
                        "Make sure you selected a **chat-capable** model "
                        f"(not an embedding model like `{EMBED_MODEL}`)."
                    )

    # Clear chat
    if st.session_state.messages:
        st.divider()
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Analytics view ────────────────────────────────────────────────────────────


def _render_analytics() -> None:
    st.markdown("## 📊 Analytics")
    stats = analytics.get_stats()

    if not stats or stats["total_queries"] == 0:
        st.info(
            "No data yet — ask some questions first, then rate the answers with 👍 / 👎."
        )
        return

    # ── Key metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", stats["total_queries"])
    c2.metric("Sessions", stats["total_sessions"])
    c3.metric("👍 Good", stats["thumbs_up"])
    c4.metric("👎 Poor", stats["thumbs_down"])

    c5, c6 = st.columns(2)
    ap = stats["approval_pct"]
    c5.metric("Approval Rate", f"{ap}%" if ap is not None else "No ratings yet")
    avg = stats["avg_best_score"]
    c6.metric(
        "Avg Best Match",
        avg if avg is not None else "—",
        help="Average L2 distance of the closest retrieved chunk. Lower = more relevant.",
    )

    # ── KB usage ──────────────────────────────────────────────────────────
    if stats["kb_usage"]:
        st.markdown("### Knowledge Base Usage")
        total_q = stats["total_queries"]
        for kb, count in stats["kb_usage"].items():
            st.progress(
                count / total_q,
                text=f"{kb}: {count} quer{'y' if count == 1 else 'ies'}",
            )

    # ── Model usage ───────────────────────────────────────────────────────
    if stats["model_usage"]:
        st.markdown("### Model Usage")
        for model, count in stats["model_usage"].items():
            st.write(f"- **{model}**: {count} quer{'y' if count == 1 else 'ies'}")

    # ── Recent queries table ──────────────────────────────────────────────
    st.markdown("### Recent Queries")
    rows = analytics.recent_queries()
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No queries recorded yet.")


# ── Entry point ───────────────────────────────────────────────────────────────

_init_state()
_render_sidebar()
if st.session_state.view == "analytics":
    _render_analytics()
else:
    _render_chat()
