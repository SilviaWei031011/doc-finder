"""英文文献反查系统 — Streamlit UI.

Run:   streamlit run app.py
Model: set DOC_FINDER_MODEL to a sentence-transformers model id or a local directory
       (default BAAI/bge-m3, downloaded from the Hugging Face Hub on first use, ~2.3 GB).
Data:  private, temporary storage per browser session; inactive data expires after ~60 minutes.
"""
import os
from contextlib import contextmanager

import streamlit as st

import core  # imports torch before faiss — see the note in core.py
import demo
from session_storage import SessionStore

st.set_page_config(page_title="英文文献反查系统", page_icon="📚", layout="wide")
st.title("📚 英文文献反查系统（中文 → 英文原文 + 页码 + 高亮）")

st.caption("[Source code (AGPL-3.0)](https://github.com/SilviaWei031011/doc-finder)")
st.info(
    "公开演示：PDF、提取文本和索引临时存于服务器，约 60 分钟无操作后自动删除，服务重启也可能清空。"
    "您可随时点击“清空我的文献”。模型在服务器本地运行，不向外部推理 API 发送文档文本或查询。"
    "托管服务商运营底层服务器，请勿上传保密或未发表材料。"
)


@st.cache_resource(show_spinner="正在加载 BGE-M3 编码模型（首次使用可能需要下载约 2.3 GB）…")
def get_model(name: str):
    return core.load_model(name)


@st.cache_resource(show_spinner=False)
def get_store():
    store = SessionStore()
    store.start_cleanup()
    return store


@st.fragment(run_every=60)
def check_session_expiry():
    # Timer reruns only this fragment: an idle open tab must not refresh its TTL.
    # A full rerun drops stale results, upload widgets and rendered-page references.
    token = st.session_state.get("session_token")
    if token and get_store().expired(token):
        st.rerun()


@contextmanager
def model_turn(where):
    status = where.empty()
    acquired = demo.MODEL_LOCK.acquire(blocking=False)
    try:
        if not acquired:
            status.info("服务器正在处理其他请求，正在等待模型空闲…")
            demo.MODEL_LOCK.acquire()
            acquired = True
        status.info("正在处理，请稍候…")
        yield
    finally:
        if acquired:
            demo.MODEL_LOCK.release()
        status.empty()


def load_model_or_stop():
    try:
        return get_model(core.model_name())
    except Exception:  # noqa: BLE001
        st.error(
            "暂时无法加载编码模型，请稍后重试。首次使用需要下载模型。"
        )
        st.stop()


def run_build(pdf_path: str, folder: str, where):
    with model_turn(where):
        model = load_model_or_stop()
        bar = where.progress(0.0, text="正在解析 PDF …")

        def cb(done, total):
            bar.progress(done / total, text=f"正在计算向量 {done}/{total} 片段")

        result = core.build_corpus(pdf_path, folder, model, progress_cb=cb,
                                   max_pages=demo.MAX_PAGES, max_chunks=demo.MAX_CHUNKS)
        bar.progress(1.0, text="索引构建完成")
        return result


def main(CORPUS_DIR, store):
    ss = st.session_state
    ss.setdefault("selected_corpus_id", None)
    ss.setdefault("results", None)
    ss.setdefault("search_seq", 0)
    ss.setdefault("flash", None)

    if ss.flash:
        st.success(ss.flash)
        ss.flash = None

    # ---------------------------------------------------------------- sidebar: library
    st.sidebar.title("📁 文献库管理")
    if st.sidebar.button("清空我的文献", key="clear_documents"):
        store.clear(ss["session_token"])
        generation = ss.get("upload_generation", 0) + 1
        ss.clear()
        ss["upload_generation"] = generation
        ss["flash"] = "已清空本会话的文献和检索结果。"
        st.rerun()
    try:
        collections = core.load_collections(CORPUS_DIR)
    except core.RegistryError:
        st.error("文献登记表读取失败，请清空本会话后重新上传。")
        st.stop()
    collections.sort(key=lambda c: c.get("created_at", 0), reverse=True)
    st.sidebar.markdown("### 📚 当前文献库")
    if not collections:
        st.sidebar.info("当前还没有任何文献，请上传。")

    for c in collections:
        with st.sidebar.expander(f"📖 {c['name']}", expanded=False):
            st.write(f"文件: `{c.get('original_filename') or c['pdf_filename']}`")
            col1, col2 = st.columns([2, 1])
            if col1.button("➡️ 选择此文献", key=f"choose_{c['id']}"):
                ss.selected_corpus_id = c["id"]
                ss.results = None
                st.rerun()
            if col2.button("🗑 删除", key=f"delete_{c['id']}"):
                try:
                    core.delete_corpus(CORPUS_DIR, c["id"])
                except Exception:  # noqa: BLE001
                    st.error("删除失败，请稍后重试。")
                else:
                    if ss.selected_corpus_id == c["id"]:
                        ss.selected_corpus_id = None
                        ss.results = None
                    ss.flash = f"已删除文献：{c['name']}"
                    st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 上传新文献（自动构建索引）")
    custom_name = st.sidebar.text_input("给这本文献起一个名称（可选，默认用文件名）", value="")
    st.sidebar.caption("每份最多 25 MB、100 页、2500 个文本片段；每个会话最多 3 份文献。仅支持文本型 PDF，无 OCR。")
    uploaded = st.sidebar.file_uploader("选择 PDF 文件（论文 / 书籍）", type=["pdf"],
                                        key=f"pdf_upload_{ss.get('upload_generation', 0)}")

    if uploaded is not None and st.sidebar.button("开始上传并建立索引", key="upload"):
        try:
            entry, result = demo.upload_document(
                CORPUS_DIR, uploaded, custom_name,
                lambda pdf, folder: run_build(pdf, folder, st.sidebar),
            )
        except (core.NoTextError, core.PDFLimitError, core.InvalidPDFError) as e:
            st.sidebar.error(f"未建立索引：{e}")
        except Exception:  # noqa: BLE001
            st.sidebar.error("上传 / 索引失败，未完成的文件已清理。请确认 PDF 有效且未加密，或稍后重试。")
        else:
            ss.selected_corpus_id = entry["id"]
            ss.results = None
            ss["upload_generation"] = ss.get("upload_generation", 0) + 1
            ss.flash = (f"上传并索引完成：{entry['name']}"
                        f"（{result.num_pages} 页，{result.num_chunks} 个片段，{result.seconds:.0f} 秒）")
            st.rerun()

    # ---------------------------------------------------------------- main: search
    selected = next((c for c in collections if c["id"] == ss.selected_corpus_id), None)
    if selected is None:
        ss.selected_corpus_id = None
        st.info("请在左侧选择或上传一本文献后使用检索功能。")
        st.stop()

    st.success(f"当前文献：**{selected['name']}**")
    folder = core.corpus_folder(CORPUS_DIR, selected["folder"])
    pdf_path = os.path.join(folder, os.path.basename(selected["pdf_filename"]))
    if not core.corpus_files_exist(folder):
        st.error("该文献的索引文件缺失，请删除后重新上传。")
        st.stop()

    with model_turn(st):
        model = load_model_or_stop()
    try:
        corpus = core.load_corpus(folder)
    except Exception:  # noqa: BLE001
        st.error("无法读取此文献的索引，请删除后重新上传。")
        st.stop()

    problem = core.compatibility_error(corpus, model)
    if problem:
        st.warning(f"⚠️ {problem}。需要用当前模型重建索引后才能检索（PDF 不会被删除）。")
        if not os.path.isfile(pdf_path):
            st.error("找不到该文献的 PDF 文件，无法重建，请删除后重新上传。")
        elif st.button("🔄 重建索引", key="rebuild"):
            try:
                result = run_build(pdf_path, folder, st)
            except (core.NoTextError, core.PDFLimitError, core.InvalidPDFError) as e:
                st.error(f"重建失败：{e}")
            except Exception:  # noqa: BLE001
                st.error("重建失败，请删除后重新上传。")
            else:
                ss.results = None
                ss.flash = f"索引已重建：{result.num_pages} 页，{result.num_chunks} 个片段，{result.seconds:.0f} 秒"
                st.rerun()
        st.stop()

    note = core.model_mismatch_note(corpus, model)
    if note:
        st.caption("此索引使用的模型标识与当前模型不同；如有检索异常，请删除后重新上传。")
    st.caption(f"共 {corpus.meta.get('num_pages', '?')} 页，{corpus.index.ntotal} 个文本片段。"
               "页码为 PDF 物理页码（从 1 起），可能与书上印刷的页码不同。")

    query = st.text_area("请输入中文译文 / 转述 / 笔记（也可直接输入英文片段）：", height=150, key="query")
    max_k = min(20, corpus.index.ntotal)
    if max_k > 1:
        top_k = st.slider("显示候选数量 Top-K", 1, max_k, min(5, max_k), 1, key="top_k")
    else:
        top_k = 1

    if st.button("开始检索", key="search"):
        if not query.strip():
            st.error("请输入内容。")
        else:
            with model_turn(st):
                try:
                    hits = core.search(corpus, model, query, top_k)
                except Exception:  # noqa: BLE001
                    st.error("暂时无法检索，请稍后重试。")
                    st.stop()
            ss.search_seq += 1
            ss.results = {"corpus_id": selected["id"], "query": query.strip(), "hits": hits, "seq": ss.search_seq}

    # Results live in session_state, so they survive the reruns triggered by the buttons below.
    res = ss.results
    if res and res["corpus_id"] == selected["id"]:
        st.markdown("### 🔍 检索结果（按相似度排序）")
        if not res["hits"]:
            st.info("没有返回结果。")
        for h in res["hits"]:
            st.markdown(f"**候选 {h.rank}（相似度 {h.score:.4f}）** — 📄 第 {h.page} 页")
            st.write(h.text)
            if st.checkbox(f"显示候选 {h.rank} 的高亮页面", key=f"hl_{res['seq']}_{h.rank}"):
                try:
                    st.image(core.render_highlight(pdf_path, h.page, h.text, max_side=1600),
                             caption=f"第 {h.page} 页（高亮）")
                except Exception:  # noqa: BLE001
                    st.error("暂时无法生成高亮页面，请稍后重试。")
            st.divider()


# The lease protects all filesystem reads and long-running builds from cleanup.
# st.stop()/st.rerun() unwind it too. No client-supplied path selects a corpus.
store = get_store()
with store.session(st.session_state.get("session_token")) as session:
    if session.reset:
        had_session = "session_token" in st.session_state
        generation = st.session_state.get("upload_generation", 0) + 1
        flash = st.session_state.get("flash")
        st.session_state.clear()
        st.session_state["upload_generation"] = generation
        st.session_state["flash"] = flash
        if had_session:
            st.info("会话文献已过期或被清理，请重新上传。")
    st.session_state["session_token"] = session.token
    check_session_expiry()
    main(session.path, store)
