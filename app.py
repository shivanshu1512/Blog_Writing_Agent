from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

from bwa_backend import app

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (only stable, non-internal selectors) ─
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, button, input, textarea, select, label, p, div {
        font-family: 'Inter', sans-serif !important;
    }

    /* App background */
    .stApp { background-color: #f5f6fa; }

    /* Stat card for live progress */
    .stat-row { display: flex; gap: 12px; margin: 12px 0; }
    .stat-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #4263eb; }
    .stat-label { font-size: 0.72rem; color: #94a3b8; margin-top: 4px;
                  text-transform: uppercase; letter-spacing: 0.6px; }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #ffffff 0%, #eef2ff 100%);
        border: 1px solid #dde4f7;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(66,99,235,0.06);
    }
    .header-card h1 { margin:0 0 6px; font-size:1.9rem; font-weight:700;
                      color:#1e293b; letter-spacing:-0.3px; }
    .header-card p  { margin:0; color:#64748b; font-size:0.92rem; }

    /* Empty state */
    .empty-state { text-align:center; padding:72px 20px; color:#94a3b8; }
    .empty-state .icon { font-size:3.5rem; margin-bottom:14px; }
    .empty-state h3 { color:#334155; font-weight:600; margin-bottom:8px; }
    .empty-state p  { font-size:0.92rem; max-width:380px; margin:0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(
    current_state: Dict[str, Any], step_payload: Any
) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(
            next(iter(step_payload.values())), dict
        ):
            current_state.update(next(iter(step_payload.values())))
        else:
            current_state.update(step_payload)
    return current_state


# ── Markdown renderer with local image support ─
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    return Path(src.strip().lstrip("./")).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md)
        return
    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        parts.append(("img", f"{(m.group('alt') or '').strip()}|||{(m.group('src') or '').strip()}"))
        last = m.end()
    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload)
            i += 1
            continue
        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))
        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or alt or None)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or alt or None)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


# ── Past blogs helpers ─────────────────────────

def list_past_blogs() -> List[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# ── Session state defaults ─────────────────────
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []


# ── Header ────────────────────────────────────
st.markdown(
    """
    <div class="header-card">
        <h1>✍️ Blog Writing Agent</h1>
        <p>Powered by LangGraph &nbsp;·&nbsp; Generate research-backed, long-form blogs in seconds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.subheader("✍️ New Blog")
    st.divider()

    topic = st.text_area(
        "Topic / Prompt",
        placeholder="e.g. How attention mechanisms work in transformers…",
        height=130,
    )
    as_of = st.date_input("As-of date", value=date.today())
    st.write("")
    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📂 Past Blogs")

    past_files = list_past_blogs()
    selected_md_file = None

    if not past_files:
        st.caption("No saved blogs found (*.md in working folder).")
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        st.write("")
        if st.button("📂 Load selected blog", use_container_width=True):
            if selected_md_file:
                md_content = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_content,
                }
                st.rerun()

    st.divider()
    st.caption("Built with LangGraph + Streamlit")


# ── Tabs ──────────────────────────────────────
tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []


# ── Run generation ────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("⏳ Running agent…", expanded=True)
    progress_placeholder = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if (
                isinstance(payload, dict)
                and len(payload) == 1
                and isinstance(next(iter(payload.values())), dict)
            ):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"→ **{node_name}**")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            n_evidence = len(current_state.get("evidence", []) or [])
            n_sections = len(current_state.get("sections", []) or [])
            n_images = len(current_state.get("image_specs", []) or [])
            plan_obj = current_state.get("plan")
            n_tasks = (
                len((plan_obj or {}).get("tasks", []))
                if isinstance(plan_obj, dict)
                else 0
            )

            progress_placeholder.markdown(
                f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="stat-value">{n_tasks}</div>
                        <div class="stat-label">Tasks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{n_evidence}</div>
                        <div class="stat-label">Evidence</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{n_sections}</div>
                        <div class="stat-label">Sections</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{n_images}</div>
                        <div class="stat-label">Images</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            logs.append(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            st.session_state["last_out"] = payload
            st.session_state["logs"].extend(logs)
            status.update(label="✅ Blog generated!", state="complete", expanded=False)
            progress_placeholder.empty()
            st.rerun()


# ── Render results ────────────────────────────
out = st.session_state.get("last_out")

if out:
    # Plan tab
    with tab_plan:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan data — this blog was loaded from a saved file.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"### {plan_dict.get('blog_title', 'Untitled')}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Audience", plan_dict.get("audience", "—"))
            c2.metric("Tone", plan_dict.get("tone", "—"))
            c3.metric("Kind", plan_dict.get("blog_kind", "—"))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "words": t.get("target_words"),
                            "research": t.get("requires_research"),
                            "citations": t.get("requires_citations"),
                            "code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)
                with st.expander("📋 Full task JSON"):
                    st.json(tasks)

    # Evidence tab
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned — closed-book mode or no search results.")
        else:
            st.markdown(f"**{len(evidence)} sources gathered**")
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append({
                    "title": e.get("title"),
                    "published": e.get("published_at"),
                    "source": e.get("source"),
                    "url": e.get("url"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Preview tab
    with tab_preview:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No markdown content found.")
        else:
            render_markdown_with_local_images(final_md)
            st.divider()

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col2:
                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Bundle (MD + images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    # Images tab
    with tab_images:
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images were generated for this blog.")
        else:
            if specs:
                with st.expander("📋 Image plan"):
                    st.json(specs)
            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ folder exists but is empty.")
                else:
                    for i in range(0, len(files), 2):
                        row_files = sorted(files)[i : i + 2]
                        cols = st.columns(2)
                        for col, img_path in zip(cols, row_files):
                            col.image(str(img_path), caption=img_path.name)
                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download All Images (.zip)",
                        data=z,
                        file_name="images.zip",
                        mime="application/zip",
                    )

    # Logs tab
    with tab_logs:
        if logs:
            st.session_state["logs"].extend(logs)
        all_logs = st.session_state.get("logs", [])
        if all_logs:
            st.text_area(
                "Event log",
                value="\n\n".join(all_logs[-80:]),
                height=520,
                label_visibility="collapsed",
            )
        else:
            st.info("No logs yet — run a generation to see live events.")

else:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">✍️</div>
            <h3>Ready to write</h3>
            <p>Enter a topic in the sidebar and click <strong>Generate Blog</strong> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
