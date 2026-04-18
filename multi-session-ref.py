"""
멀티세션 RAG 챗봇 — Supabase(세션·벡터) + OpenAI(gpt-4o-mini, 임베딩) + Streamlit
실행 전 Supabase에 multi-session-ref.sql 을 적용하세요.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# -----------------------------------------------------------------------------
# 경로: AI-Education/.env (실행 위치와 무관)
# -----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _ROOT / ".env"
load_dotenv(_ENV_PATH)

# -----------------------------------------------------------------------------
# 로깅 (ref.txt: ERROR/WARNING만, HTTP 로그 억제)
# Streamlit Cloud 등에서는 프로젝트 루트에 쓰기 불가 → /tmp 등으로 폴백
# -----------------------------------------------------------------------------
def _resolve_log_dir() -> Path:
    preferred = _ROOT / "logs"
    try:
        preferred.mkdir(exist_ok=True)
        return preferred
    except (PermissionError, OSError):
        fallback = Path(tempfile.gettempdir()) / "ai-education-chatbot-logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


_LOG_DIR = _resolve_log_dir()
_LOG_FILE = _LOG_DIR / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"

for _name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.WARNING)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(_LOG_FILE) for h in _root_logger.handlers):
    try:
        _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        _fh.setLevel(logging.WARNING)
        _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _root_logger.addHandler(_fh)
    except (PermissionError, OSError):
        pass


def _log_exc(msg: str, exc: BaseException) -> None:
    logging.getLogger(__name__).warning("%s: %s", msg, exc, exc_info=True)


# -----------------------------------------------------------------------------
# 환경 변수
# -----------------------------------------------------------------------------
def _env_ok() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SUPABASE_URL"):
        missing.append("SUPABASE_URL")
    if not os.getenv("SUPABASE_ANON_KEY"):
        missing.append("SUPABASE_ANON_KEY")
    return (len(missing) == 0, missing)


def _get_supabase():
    try:
        from supabase import create_client
    except ImportError:
        return None
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


# -----------------------------------------------------------------------------
# ref.txt 유틸
# -----------------------------------------------------------------------------
def remove_separators(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"~~[^~]*~~", "", text)
    text = re.sub(r"^[\t ]*[-=]{3,}[\t ]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -----------------------------------------------------------------------------
# OpenAI
# -----------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VECTOR_BATCH = 10
RAG_TOP_K = 10


def get_openai_client() -> OpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        for d in sorted(resp.data, key=lambda x: x.index):
            out.append(list(d.embedding))
    return out


def generate_session_title(client: OpenAI, first_q: str, first_a: str) -> str:
    prompt = (
        "다음은 사용자의 첫 질문과 챗봇의 첫 답변입니다. "
        "이 대화를 한 줄로 요약하는 짧은 세션 제목(최대 40자, 한국어)만 출력하세요. "
        "따옴표나 부가 설명 없이 제목만.\n\n"
        f"[질문]\n{first_q[:2000]}\n\n[답변]\n{first_a[:2000]}"
    )
    r = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    title = (r.choices[0].message.content or "새 세션").strip().split("\n")[0][:80]
    return title or "새 세션"


def generate_followup_questions(client: OpenAI, question: str, answer: str) -> str:
    prompt = (
        "사용자가 문서 RAG 챗봇을 사용 중입니다. 아래 질문과 답변을 바탕으로 "
        "이어서 물어보면 좋은 질문을 정확히 3개만 한국어로 제시하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        f"\n[질문]\n{question[:1500]}\n\n[답변]\n{answer[:4000]}"
    )
    r = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}],
    )
    return (r.choices[0].message.content or "").strip()


def stream_chat_answer(
    client: OpenAI,
    messages: list[dict[str, str]],
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    for ev in stream:
        if ev.choices and ev.choices[0].delta.content:
            yield ev.choices[0].delta.content


# -----------------------------------------------------------------------------
# Supabase — 세션 / 메시지 / 벡터
# -----------------------------------------------------------------------------
def db_insert_session(supabase: Any, title: str = "새 세션", sid: str | None = None) -> str:
    row: dict[str, Any] = {"title": title}
    if sid:
        row["id"] = sid
    r = supabase.table("chat_sessions").insert(row).execute()
    if not r.data:
        raise RuntimeError("세션 INSERT 실패")
    return str(r.data[0]["id"])


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_update_session_title(supabase: Any, session_id: str, title: str) -> None:
    supabase.table("chat_sessions").update({"title": title, "updated_at": _utc_iso()}).eq("id", session_id).execute()


def db_list_sessions(supabase: Any) -> list[dict[str, Any]]:
    r = (
        supabase.table("chat_sessions")
        .select("id,title,updated_at")
        .order("updated_at", desc=True)
        .execute()
    )
    return list(r.data or [])


def db_delete_session(supabase: Any, session_id: str) -> None:
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()


def db_replace_messages(supabase: Any, session_id: str, chat_history: list[dict[str, str]]) -> None:
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows = []
    for i, m in enumerate(chat_history):
        rows.append(
            {
                "session_id": session_id,
                "msg_index": i,
                "role": m["role"],
                "content": m["content"],
            }
        )
    if rows:
        supabase.table("chat_messages").insert(rows).execute()
    supabase.table("chat_sessions").update({"updated_at": _utc_iso()}).eq("id", session_id).execute()


def db_load_messages(supabase: Any, session_id: str) -> list[dict[str, str]]:
    r = (
        supabase.table("chat_messages")
        .select("msg_index,role,content")
        .eq("session_id", session_id)
        .order("msg_index")
        .execute()
    )
    out: list[dict[str, str]] = []
    for row in r.data or []:
        out.append({"role": row["role"], "content": row["content"]})
    return out


def _embedding_to_db_value(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def db_insert_vector_batch(
    supabase: Any,
    session_id: str,
    chunks: list[tuple[str, str, list[float]]],
) -> None:
    """chunks: (content, file_name, embedding)"""
    rows = []
    for content, file_name, emb in chunks:
        rows.append(
            {
                "session_id": session_id,
                "content": content,
                "file_name": file_name,
                "embedding": _embedding_to_db_value(emb),
                "metadata": {},
            }
        )
    for i in range(0, len(rows), VECTOR_BATCH):
        supabase.table("vector_documents").insert(rows[i : i + VECTOR_BATCH]).execute()


def db_copy_vectors_to_session(supabase: Any, from_sid: str, to_sid: str) -> None:
    r = supabase.table("vector_documents").select("content,file_name,embedding").eq("session_id", from_sid).execute()
    data = r.data or []
    if not data:
        return
    rows = []
    for row in data:
        emb = row.get("embedding")
        if isinstance(emb, list):
            emb_s = _embedding_to_db_value([float(x) for x in emb])
        else:
            emb_s = str(emb)
        rows.append(
            {
                "session_id": to_sid,
                "content": row["content"],
                "file_name": row["file_name"],
                "embedding": emb_s,
                "metadata": {},
            }
        )
    for i in range(0, len(rows), VECTOR_BATCH):
        supabase.table("vector_documents").insert(rows[i : i + VECTOR_BATCH]).execute()


def db_distinct_vector_filenames(supabase: Any, session_id: str) -> list[str]:
    r = supabase.table("vector_documents").select("file_name").eq("session_id", session_id).execute()
    names = sorted({row["file_name"] for row in (r.data or []) if row.get("file_name")})
    return names


def retrieve_context_rpc(
    supabase: Any,
    client: OpenAI,
    session_id: str,
    query: str,
) -> str:
    qemb = embed_texts(client, [query])[0]
    emb_candidates: list[Any] = [qemb, _embedding_to_db_value(qemb)]
    for emb_payload in emb_candidates:
        try:
            r = supabase.rpc(
                "match_vector_documents",
                {
                    "query_embedding": emb_payload,
                    "match_count": RAG_TOP_K,
                    "filter_session_id": session_id,
                },
            ).execute()
            parts: list[str] = []
            for row in r.data or []:
                fn = row.get("file_name", "")
                ct = row.get("content", "")
                parts.append(f"[파일: {fn}]\n{ct}")
            return "\n\n".join(parts)
        except Exception as exc:
            _log_exc("match_vector_documents RPC 시도 실패", exc)
            continue
    try:
        r2 = supabase.table("vector_documents").select("content,file_name").eq("session_id", session_id).execute()
        docs = r2.data or []
        if not docs:
            return ""
        qlow = query.lower()
        scored = []
        for d in docs:
            c = (d.get("content") or "").lower()
            score = sum(1 for w in qlow.split() if len(w) > 1 and w in c)
            scored.append((score, d))
        scored.sort(key=lambda x: -x[0])
        picked = [d for _, d in scored[:RAG_TOP_K]]
        return "\n\n".join(f"[파일: {d.get('file_name','')}]\n{d.get('content','')}" for d in picked)
    except Exception as exc2:
        _log_exc("벡터 폴백 조회 실패", exc2)
        return ""


# -----------------------------------------------------------------------------
# PDF 처리 (파일명을 청크에 명시 — file_name NULL 방지)
# -----------------------------------------------------------------------------
def process_pdf_files(
    paths: list[tuple[str, str]],
    supabase: Any,
    client: OpenAI,
    session_id: str,
) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks: list[tuple[str, str]] = []
    for display_name, fs_path in paths:
        loader = PyPDFLoader(fs_path)
        pages = loader.load()
        for p in pages:
            p.metadata = dict(p.metadata or {})
            p.metadata["file_name"] = display_name
        subdocs = splitter.split_documents(pages)
        for d in subdocs:
            fn = d.metadata.get("file_name") or display_name
            all_chunks.append((d.page_content, str(fn)))

    if not all_chunks:
        return 0

    texts = [c[0] for c in all_chunks]
    embeddings = embed_texts(client, texts)
    insert_payload: list[tuple[str, str, list[float]]] = []
    for (content, fn), emb in zip(all_chunks, embeddings):
        if len(emb) != EMBED_DIM:
            raise ValueError(f"임베딩 차원 불일치: 기대 {EMBED_DIM}, 실제 {len(emb)}")
        insert_payload.append((content, fn, emb))

    for i in range(0, len(insert_payload), VECTOR_BATCH):
        db_insert_vector_batch(supabase, session_id, insert_payload[i : i + VECTOR_BATCH])
    return len(insert_payload)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def _init_session_state(supabase: Any | None) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "processed_file_names" not in st.session_state:
        st.session_state.processed_file_names = []
    if "vector_db_modal" not in st.session_state:
        st.session_state.vector_db_modal = False
    if "pending_auto_title" not in st.session_state:
        st.session_state.pending_auto_title = False
    if "current_session_id" in st.session_state:
        return

    if supabase is None:
        st.session_state.current_session_id = str(uuid.uuid4())
        return

    try:
        rows = db_list_sessions(supabase)
        if rows:
            latest = str(rows[0]["id"])
            st.session_state.current_session_id = latest
            st.session_state.chat_history = db_load_messages(supabase, latest)
            st.session_state.pending_auto_title = len(st.session_state.chat_history) >= 2
        else:
            sid = str(uuid.uuid4())
            db_insert_session(supabase, "새 세션", sid=sid)
            st.session_state.current_session_id = sid
            st.session_state.pending_auto_title = False
    except Exception as exc:
        _log_exc("초기 세션 로드/생성 실패", exc)
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.pending_auto_title = False


def _apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        div.stButton > button:first-child {
            background-color: #ff69b4;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    logo_path = _ROOT / "logo.png"
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if logo_path.is_file():
            st.image(str(logo_path), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
            <div style="text-align:center;">
            <span style="font-size:4rem !important; font-weight:700; color:#1f77b4 !important;">멀티세션 RAG</span>
            <span style="font-size:4rem !important; font-weight:700; color:#ffd700 !important;"> 챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


def _persist_auto(supabase: Any, oai: OpenAI) -> None:
    sid = st.session_state.get("current_session_id")
    if not sid or not supabase:
        return
    try:
        db_replace_messages(supabase, sid, st.session_state.chat_history)
    except Exception as exc:
        _log_exc("자동 저장 실패", exc)


def _load_session_into_ui(supabase: Any, session_id: str) -> None:
    try:
        msgs = db_load_messages(supabase, session_id)
        st.session_state.chat_history = msgs
        st.session_state.current_session_id = session_id
        st.session_state.processed_file_names = db_distinct_vector_filenames(supabase, session_id)
        st.session_state.pending_auto_title = True
    except Exception as exc:
        _log_exc("세션 로드 실패", exc)
        st.error(f"세션을 불러오지 못했습니다: {exc}")


def main() -> None:
    st.set_page_config(page_title="멀티세션 RAG 챗봇", page_icon="📚", layout="wide")
    _apply_custom_css()

    ok, missing = _env_ok()
    supabase = _get_supabase() if ok else None
    oai = get_openai_client() if ok else None

    if not ok:
        st.warning("다음 환경 변수가 `.env`에 없습니다: " + ", ".join(missing))
        st.caption(f"로드한 env 경로: {_ENV_PATH}")
    elif supabase is None:
        st.warning("supabase 패키지가 없거나 URL/키가 올바르지 않습니다. `pip install supabase` 후 다시 시도하세요.")
    elif oai is None:
        st.warning("OPENAI_API_KEY가 없어 LLM을 사용할 수 없습니다.")

    _init_session_state(supabase)

    _render_header()

    # --- 사이드바
    with st.sidebar:
        st.markdown("#### 세션 관리")
        sessions = db_list_sessions(supabase) if supabase else []
        id_to_title = {s["id"]: s.get("title") or "제목 없음" for s in sessions}
        options = [s["id"] for s in sessions]

        def _on_pick_change() -> None:
            if not supabase:
                return
            sel = st.session_state.get("sb_session_pick")
            if sel:
                _load_session_into_ui(supabase, sel)

        pick = st.selectbox(
            "저장된 세션",
            options=options if options else [""],
            format_func=lambda x: id_to_title.get(x, "—") if x else "(세션 없음)",
            key="sb_session_pick",
            index=options.index(st.session_state["current_session_id"])
            if st.session_state.get("current_session_id") in options
            else 0,
            on_change=_on_pick_change,
            disabled=not supabase or not options,
        )

        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("세션저장", use_container_width=True) and supabase and oai:
                hist = st.session_state.chat_history
                if len(hist) < 2:
                    st.warning("저장할 대화(질문·답변)가 충분하지 않습니다.")
                else:
                    try:
                        first_u = next((m["content"] for m in hist if m["role"] == "user"), "")
                        first_a = next((m["content"] for m in hist if m["role"] == "assistant"), "")
                        title = generate_session_title(oai, first_u, first_a)
                        new_id = str(uuid.uuid4())
                        db_insert_session(supabase, title, sid=new_id)
                        db_replace_messages(supabase, new_id, hist)
                        old_sid = st.session_state.current_session_id
                        db_copy_vectors_to_session(supabase, old_sid, new_id)
                        st.success(f"새 세션으로 저장했습니다: {title}")
                    except Exception as exc:
                        _log_exc("세션저장 실패", exc)
                        st.error(str(exc))

        with bcol2:
            if st.button("세션로드", use_container_width=True) and supabase and pick:
                _load_session_into_ui(supabase, pick)
                st.success("세션을 불러왔습니다.")

        if st.button("세션삭제", use_container_width=True) and supabase:
            target = pick or st.session_state.get("current_session_id")
            if not target:
                st.warning("삭제할 세션을 선택하세요.")
            else:
                try:
                    db_delete_session(supabase, target)
                    if st.session_state.get("current_session_id") == target:
                        new_sid = str(uuid.uuid4())
                        db_insert_session(supabase, "새 세션", sid=new_sid)
                        st.session_state.current_session_id = new_sid
                        st.session_state.chat_history = []
                        st.session_state.processed_file_names = []
                        st.session_state.pending_auto_title = False
                    st.success("세션을 삭제했습니다.")
                    st.rerun()
                except Exception as exc:
                    _log_exc("세션삭제 실패", exc)
                    st.error(str(exc))

        if st.button("화면초기화", use_container_width=True) and supabase and oai:
            st.session_state.chat_history = []
            st.session_state.processed_file_names = []
            st.session_state.pending_auto_title = False
            st.session_state.uploader_key += 1
            try:
                new_sid = str(uuid.uuid4())
                db_insert_session(supabase, "새 세션", sid=new_sid)
                st.session_state.current_session_id = new_sid
            except Exception as exc:
                _log_exc("화면초기화 중 세션 생성 실패", exc)
            st.rerun()

        if st.button("vectordb", use_container_width=True) and supabase:
            st.session_state.vector_db_modal = True

        st.markdown("#### LLM")
        st.radio("모델", options=[LLM_MODEL], index=0, disabled=True)

        st.markdown("#### RAG (PDF)")
        uploaded = st.file_uploader(
            "PDF 업로드 (다중)",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_up_{st.session_state.uploader_key}",
        )
        if st.button("파일 처리하기") and supabase and oai:
            if not uploaded:
                st.warning("PDF 파일을 선택하세요.")
            else:
                tmp_paths: list[tuple[str, str]] = []
                try:
                    for uf in uploaded:
                        suffix = Path(uf.name).suffix or ".pdf"
                        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tf.write(uf.getvalue())
                        tf.close()
                        tmp_paths.append((uf.name, tf.name))
                    n = process_pdf_files(tmp_paths, supabase, oai, st.session_state.current_session_id)
                    names = list({p[0] for p in tmp_paths})
                    st.session_state.processed_file_names = sorted(
                        set(st.session_state.processed_file_names) | set(names)
                    )
                    st.success(f"벡터 DB에 {n}개 청크를 저장했습니다.")
                    _persist_auto(supabase, oai)
                except Exception as exc:
                    _log_exc("PDF 처리 실패", exc)
                    st.error(str(exc))
                finally:
                    for _, pth in tmp_paths:
                        try:
                            os.unlink(pth)
                        except OSError:
                            pass

        st.text(
            "현재 설정\n"
            f"- 모델: {LLM_MODEL}\n"
            f"- 임베딩: {EMBED_MODEL} ({EMBED_DIM}차원)\n"
            f"- 세션 ID: {st.session_state.get('current_session_id','')}\n"
            f"- 처리된 파일(이름): {len(st.session_state.processed_file_names)}개\n"
            f"- 대화 메시지 수: {len(st.session_state.chat_history)}"
        )

    if st.session_state.get("vector_db_modal") and supabase:
        with st.expander("Vector DB — 현재 세션에 저장된 파일명", expanded=True):
            sid = st.session_state.current_session_id
            try:
                files = db_distinct_vector_filenames(supabase, sid)
                if files:
                    st.markdown("\n".join(f"- {n}" for n in files))
                else:
                    st.info("이 세션에 저장된 벡터 문서가 없습니다.")
            except Exception as exc:
                st.error(str(exc))
            if st.button("닫기", key="btn_close_vectordb"):
                st.session_state.vector_db_modal = False
                st.rerun()

    # --- 채팅 표시
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(remove_separators(m["content"]), unsafe_allow_html=True)

    if not oai or not supabase:
        st.stop()

    user_q = st.chat_input("질문을 입력하세요")
    if not user_q:
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(remove_separators(user_q))

    ctx = retrieve_context_rpc(supabase, oai, st.session_state.current_session_id, user_q)
    sys_prompt = (
        "당신은 문서 기반 도우미입니다. 제공된 참고 문맥을 우선 활용하고, "
        "답변은 # ## ### 마크다운 헤딩으로 구조화하며 한국어 존댓말을 사용합니다. "
        "구분선(--- 등)과 취소선은 사용하지 마세요."
    )
    ctx_block = f"[참고 문맥]\n{ctx}\n\n" if ctx else ""
    mem_lines = []
    for m in st.session_state.chat_history[-50:]:
        mem_lines.append(f"{m['role']}: {m['content'][:2000]}")
    mem = "\n".join(mem_lines)

    api_messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt + "\n이전 대화 요약:\n" + mem},
        {"role": "user", "content": ctx_block + "[질문]\n" + user_q},
    ]

    assistant_text = ""
    with st.chat_message("assistant"):
        ph = st.empty()
        try:
            for piece in stream_chat_answer(oai, api_messages):
                assistant_text += piece
                ph.markdown(remove_separators(assistant_text) + "▌", unsafe_allow_html=True)
            tail = generate_followup_questions(oai, user_q, assistant_text)
            extra = "\n\n### 💡 다음에 물어볼 수 있는 질문들\n" + tail
            assistant_text += extra
            ph.markdown(remove_separators(assistant_text), unsafe_allow_html=True)
        except Exception as exc:
            _log_exc("답변 스트리밍 실패", exc)
            ph.error(str(exc))
            assistant_text = "(오류로 답변을 완성하지 못했습니다.)"

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

    # 첫 턴 이후 세션 제목 자동 갱신
    if len(st.session_state.chat_history) >= 2 and not st.session_state.pending_auto_title:
        try:
            u0 = st.session_state.chat_history[0]["content"]
            a0 = st.session_state.chat_history[1]["content"]
            t = generate_session_title(oai, u0, a0)
            db_update_session_title(supabase, st.session_state.current_session_id, t)
            st.session_state.pending_auto_title = True
        except Exception as exc:
            _log_exc("제목 자동 생성 실패", exc)

    _persist_auto(supabase, oai)


if __name__ == "__main__":
    main()
